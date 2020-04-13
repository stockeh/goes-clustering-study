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
import torch.nn as nn
import torch.distributed as dist

from torchvision import transforms

import visualizations as viz
import goesdataset as gd
import models

best_rmse = float("inf")

def main():
    parser = argparse.ArgumentParser(description="Distributed PyTorch")

    parser.add_argument("world_size", type=int, default=4,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""")

    parser.add_argument("rank", type=int, default=None,
        help="Global rank of this process. Pass in 0 for master.")

    parser.add_argument('--dist-conn', default='bentley:15516', type=str,
                        help='master host:port used to set up distributed training')

    parser.add_argument('-b', '--batch-size', type=int, default=25,
        metavar='N',
        help='mini-batch size (default: 3).')

    parser.add_argument('--epochs', default=1, type=int, metavar='N',
        help='number of total epochs to run')

    parser.add_argument("--data", type=str, default="../data/",
        help="Directory containing the data to be run on.")

    parser.add_argument('--cuda', type=bool, default=False,
        help='True to use CUDA on GPU. WARN: only implemented on CPU!.')

    parser.add_argument('-d', '--latent-dim', type=int, default=10,
        help='Dimensionality of the latent vector.')

    parser.add_argument('-c', '--channels', type=int, nargs='+', default=[0],
        help='Channels to evaluate (use at end).')

    args = parser.parse_args()

    args.distributed = args.world_size > 1
    if args.cuda and not torch.cuda.is_available():
        print('WARN: CUDA is not available!')
        args.cuda = False

    assert args.rank is not None, 'must provide rank argument.'
    assert args.world_size > args.rank, 'world_size must be greater than rank.'
    assert os.path.exists(args.data) and 'hdfs' not in args.data, 'data directory must exist on disk.'
    assert len(args.channels) > 0, f'{args.channels} :: must provide valid channels, e.g., [0, 2, 3].'

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
    train_set = gd.GOESDataset(root_dir=args.data, channels=True,
                               transform=transforms.Compose([
                                    gd.Square(), gd.Normalize(), gd.ToTensor()
                               ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=4, sampler=train_sampler, collate_fn=gd.custom_collate)

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
    but needs additional logic to set the best_rmse and start_epoch.
    """
    checkpoint = torch.load(args.resume)
    last_epoch = checkpoint['epoch']
    best_rmse = checkpoint['best_rmse']
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
        print(f'INFO: Saving best model to {best_filename} with lowest '
               'error \u03BC = {0:.3f} from node rank: {1}'.format(
               state['best_rmse'], dist.get_rank()))

        shutil.copyfile(filename, best_filename)


def flatten_dimension(train_loader,args):
    shape = train_loader.dataset[0][args.channels, ...].shape
    return np.product(shape)


def main_worker(args):
    global best_rmse

    train_loader, train_sampler = load_dataset(args)
    nnet = models.AutoencoderLinear(flatten_dimension(train_loader, args),
                                    args.latent_dim)
    print(nnet)

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

    # optimizer = torch.optim.SGD(nnet.parameters(), lr=0.01, momentum=0.50)
    optimizer = torch.optim.Adam(nnet.parameters(), lr=0.01,
                             betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        progress = train(train_loader, nnet, criterion, optimizer, epoch, args)
        # progress.display(len(train_loader))

        rmse = progress.getAvgMeter('Error').avg

        # remember best error and save checkpoint
        # is_best = rmse < best_rmse
        # best_rmse = min(rmse, best_rmse)
        #
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': nnet.state_dict(),
        #     'best_rmse' : best_rmse,
        #     'optimizer' : optimizer.state_dict(),
        # }, is_best)

    if dist.get_rank() == 0:
        print('HERE')
        viz.visualize(nnet, train_loader, args.channels[0])


def train(train_loader, nnet, criterion, optimizer, epoch, args):
    n_batches  = len(train_loader)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time  = AverageMeter('Data', ':6.3f')
    error      = AverageMeter('Error', ':.4')
    progress   = ProgressMeter(n_batches, [batch_time, data_time, error],
                            prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    nnet.train()

    end = time.time()
    for i, X in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        X = X[:, args.channels, ...]
        X = X.reshape(X.shape[0], np.product(X.shape[1:]))
        if args.cuda:
            X = X.cuda(non_blocking=True)

        Y = nnet(X)
        # RMSE error
        rmse = torch.sqrt(criterion(Y, X))

        error.update(rmse.item(), X.shape[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        rmse.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # display progress
        # printProgressBar(i + 1, n_batches, prefix='Progress:', suffix='Complete', length=50)
        progress.display(i + 1)

    return progress


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
        print('\t'.join(entries))

    def getAvgMeter(self, name):
        for meter in self.meters:
            if name in meter.name:
                return meter
        return AverageMeter('None', ':.3f')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
