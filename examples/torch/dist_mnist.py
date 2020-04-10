import os
import sys
import traceback
import datetime
import socket
import random
import shutil
import time
import argparse
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

best_acc1 = 0

def main():
    parser = argparse.ArgumentParser(description="Distributed PyTorch")

    parser.add_argument("world_size", type=int, default=4,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""")

    parser.add_argument("rank", type=int, default=None,
        help="Global rank of this process. Pass in 0 for master.")

    parser.add_argument('--dist-conn', default='mars:15516', type=str,
                        help='host:port used to set up distributed training')

    parser.add_argument('-b', '--batch-size', type=int, default=128,
        metavar='N',
        help='mini-batch size (default: 128).')

    parser.add_argument('--epochs', default=1, type=int, metavar='N',
        help='number of total epochs to run')

    parser.add_argument("--data", type=str, default="./data",
        help="Directory containing the data to be run on.")

    parser.add_argument('--cuda', type=bool, default=False,
        help='True to use CUDA on GPU.')

    args = parser.parse_args()

    args.distributed = args.world_size > 1
    if args.cuda and not torch.cuda.is_available():
        print('WARN: CUDA is not available!')
        args.cuda = False

    assert args.rank is not None, 'must provide rank argument.'
    assert args.world_size > args.rank, 'world_size must be greater than rank.'
    assert os.path.exists(args.data) and 'hdfs' not in args.data, 'data directory must exist on disk.'

    try:
        setup(args)
        start = time.time()
        main_worker(args)
        print('total: {:.3f} (s)'.format(time.time() - start))
    except Exception as e:
        traceback.print_exc()
        sys.exit(3)


def setup(args):
    # init process group
    dist.init_process_group('nccl' if args.cuda else 'gloo', rank=args.rank,
            world_size=args.world_size, init_method='tcp://' + args.dist_conn,
            timeout=datetime.timedelta(weeks=120))

    # explicitly set seed for all models to start with same w
    torch.manual_seed(42)

    print(socket.gethostname() + ': setup completed!')


def load_dataset(args):
    train_set = datasets.MNIST(args.data, train=True, download=True,
                                transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler)

    return train_loader, train_sampler


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1,
                      length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    if iteration == total:
        print()


def load_model(nnet, optimizer, args):
    r"""Currently NOT being used. This could be used for checkpointing,
    but needs additional logic to set the best_acc1 and start_epoch.
    """
    checkpoint = torch.load(args.resume)
    last_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']
    nnet.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return nnet, optimizer


def save_checkpoint(state, is_best, filename='checkpoint'):
    r"""All nodes will save an update to `checkpoint`. Therefore, the node to
    finish last will have the most recent checkpoint. If the checkpoint is the
    best model, then it will save save.
    """
    date = datetime.datetime.now().strftime('%m%d%Y')
    filename = filename + '_' + date + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        best_filename = 'model_best' + '_' + date + '.pth.tar'
        print(f'INFO: Saving best model to {best_filename} with top '
               'acc \u03BC = {0:.3f} from node rank: {1}'.format(
               state['best_acc1'].item(),dist.get_rank()))

        shutil.copyfile(filename, best_filename)


def main_worker(args):
    global best_acc1

    train_loader, train_sampler = load_dataset(args)
    nnet = Net()

    if args.distributed:
        if args.cuda:
            nnet = nnet.cuda()
        r"""This container parallelizes the application of the given module by
        splitting the input across the specified devices by chunking in the batch
        dimension. The module is replicated on each machine and each device, and
        each such replica handles a portion of the input. During the backwards
        pass, gradients from each node are averaged.
        """
        nnet = nn.parallel.DistributedDataParallel(nnet).float()

    optimizer = optim.SGD(nnet.parameters(), lr=0.01, momentum=0.50)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        progress = train(train_loader, nnet, criterion, optimizer, epoch, args)
        progress.display(len(train_loader))

        # WARNING: TOP ACCURACY FOR TRAINING SET --NOT TESTING!!--
        acc1 = progress.getAvgMeter('Acc@1').avg

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': nnet.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, nnet, criterion, optimizer, epoch, args):
    n_batches  = len(train_loader)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time  = AverageMeter('Data', ':6.3f')
    losses     = AverageMeter('Loss', ':.4e')
    top1       = AverageMeter('Acc@1', ':6.2f')
    top5       = AverageMeter('Acc@5', ':6.2f')
    progress   = ProgressMeter(n_batches, [batch_time, data_time, losses, top1, top5],
                            prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    nnet.train()

    end = time.time()
    for i, (X, T) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            X = X.cuda(non_blocking=True)
            T = T.cuda(non_blocking=True)

        # compute output
        Y = nnet(X)
        loss = criterion(Y, T)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(Y, T, topk=(1, 5))

        losses.update(loss.item(), X.size(0))
        top1.update(acc1[0], X.size(0))
        top5.update(acc5[0], X.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        printProgressBar(i + 1, n_batches, prefix='Progress:', suffix='Complete', length=50)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO: display interations (need to add args.print_freq to work)
        # if i % args.print_freq == 0:
        #     progress.display(i)

    return progress


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: v\u208B\u2081={val' + self.fmt + '} (\u03BC={avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        items = len(entries)
        print('\t'.join(entries[:items // 2]))
        print('\t'.join(entries[items//2:]))

    def getAvgMeter(self, name):
        for meter in self.meters:
            if name in meter.name:
                return meter
        return AverageMeter('None', ':.3f')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, X):
        Z = F.relu(self.conv1(X))
        Z = F.max_pool2d(Z, 2, 2)
        Z = F.relu(self.conv2(Z))
        Z = F.max_pool2d(Z, 2, 2)
        Z = Z.view(-1, 4*4*50)
        Z = F.relu(self.fc1(Z))
        return self.fc2(Z)


if __name__ == '__main__':
    main()
