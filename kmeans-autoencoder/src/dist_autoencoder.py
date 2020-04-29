import os
import sys
import traceback
import datetime
import socket
import random
import shutil
import time
import argparse
import itertools
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.distributed as dist

from torchvision import transforms
from contextlib import nullcontext

from metering import AverageMeter, ProgressMeter
import visualizations as viz
import goesdataset as gd
import models

best_loss = float("inf")
best_nnet = None
best_result_index = 0

def main():
    parser = argparse.ArgumentParser(description="Distributed PyTorch")

    parser.add_argument('world_size', type=int, default=4, metavar='N',
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""")

    parser.add_argument('rank', type=int, default=None, metavar='N',
        help="Global rank of this process. Pass in 0 for master.")

    parser.add_argument('--dist-conn', type=str, default='bentley:15516',
                        help='master host:port used to set up distributed training')

    parser.add_argument("--data", type=str, default="../data/", metavar='N',
        help="Directory containing the data to be run on.")

    parser.add_argument('--cuda', type=bool, default=False, metavar='N',
        help='True to use CUDA on GPU. WARN: only implemented on CPU!.')

    parser.add_argument('--model', type=str, default='cnn', metavar='N',
        help='Type of model to analyze either, cnn | lin | vae.')

    args = parser.parse_args()

    args.distributed = args.world_size > 1
    if args.cuda and not torch.cuda.is_available():
        print('WARN: CUDA is not available!')
        args.cuda = False

    assert args.rank is not None, 'must provide rank argument.'
    assert args.world_size > args.rank, 'world_size must be greater than rank.'
    if 'hdfs' not in args.data:
        assert os.path.exists(args.data), 'data directory must exist on disk.'
    assert args.model in ['cnn', 'lin','vae'], 'must provide a valid --model type.'

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


def load_dataset(args, distributed=True):
    train_set = gd.GOESDataset(root_dir=args.data,
                               transform=transforms.Compose([
                                    gd.Square(), gd.ToTensor(),
                                    transforms.Normalize(mean=[0.5]*16, std=[0.5]*16)
                               ]))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=12, sampler=train_sampler, collate_fn=gd.custom_collate)

    return train_loader, train_sampler


def main_worker(args):
    global best_loss, best_nnet, best_result_index

    results = pd.DataFrame(columns=['world_size', 'model', 'algo', 'rho', 'epochs', 'batch_size', 'channels',
                                    'latent_dim', 'hidden_dims', 'ker_str_pad', 'error_trace',
                                    'time_data_trace', 'time_step_trace', 'duration'])
    algo    = 'adam'
    l_epoch = [8]
    l_rho   = [0.001]
    l_batch_size = [512]
    l_latent_dim = [3]
    l_hidden_dims = [[10, 10, 10]]
    l_ker_str_pad = [[(3, 1, 0), (4, 2, 0), (4, 2, 0)]]
    l_channels = [[1, 7, 12]]

    for (epochs, rho, batch_size, latent_dim, hidden_dims, ker_str_pad, channels) in itertools.product(
        l_epoch, l_rho, l_batch_size, l_latent_dim, l_hidden_dims, l_ker_str_pad, l_channels):

        args.hidden_dims, args.ker_str_pad, args.latent_dim = hidden_dims, ker_str_pad, latent_dim
        args.batch_size = batch_size
        args.channels = channels
        train_loader, train_sampler = load_dataset(args)
        nnet = models.get_model(train_loader.dataset[0][0][args.channels, ...].shape, args)

        if args.distributed:
            if args.cuda:
                nnet = nnet.cuda()
            nnet = nn.parallel.DistributedDataParallel(nnet).float()
        print(nnet)
        if algo is 'sgd':
            optimizer = torch.optim.SGD(nnet.parameters(), lr=rho, momentum=0.50)
        else:
            optimizer = torch.optim.Adam(nnet.parameters(), lr=rho,
                                 betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        criterion = nn.MSELoss()

        start = time.time()
        trace = {'error': [], 'data': [], 'step': []}
        for epoch in range(epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            # train for one epoch
            progress = train(train_loader, nnet, criterion,
                             optimizer, epoch, trace, args)

        results.loc[len(results)] = [args.world_size, args.model, algo, rho, epochs, args.batch_size, args.channels,
                                     latent_dim, hidden_dims, ker_str_pad, trace['error'],
                                     trace['data'], trace['step'], time.time() - start]

        loss = progress.getAvgMeter('Error').avg

        # remember best error and save nnet
        if loss < best_loss:
            best_loss = loss
            best_nnet = nnet
            best_result_index = len(results) - 1

        if dist.get_rank() == 0 and False:
            dir = f'../results/experiment-{len(results)}/'
            os.makedirs(os.path.dirname(dir), exist_ok=True)
            results.loc[len(results) - 1].to_csv(dir + 'model.csv', header=False)
            full_train_loader, _ = load_dataset(args, distributed=False)
            if args.distributed:
                nnet = nnet.module
            save_latent_vector(nnet, full_train_loader, dir + 'latent_vectors.csv', args)

    if dist.get_rank() == 0:
        if args.distributed:
            best_nnet = best_nnet.module
        save_results(best_nnet, optimizer, results, best_result_index, args)


def train(train_loader, nnet, criterion, optimizer, epoch, trace, args):
    n_batches  = len(train_loader)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time  = AverageMeter('Data', ':6.3f')
    error      = AverageMeter('Error', ':.4')
    progress   = ProgressMeter(n_batches, [batch_time, data_time, error],
                            prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    nnet.train()

    end = time.time()
    for i, (X,_,_) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        trace['data'].append(data_time.val)

        X = X[:, args.channels, ...]

        if 'lin' in args.model:
            X = X.reshape(X.shape[0], np.product(X.shape[1:]))

        if args.cuda:
            X = X.cuda(non_blocking=True)

        if 'vae' in args.model:
            results = nnet(X)
            loss = nnet.loss_function(*results)
        else:
            Y = nnet(X)
            loss = torch.sqrt(criterion(Y, X))

        trace['error'].append(loss.item())
        error.update(loss.item(), X.shape[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        trace['step'].append(batch_time.val)
        end = time.time()

        # display progress
        progress.display(i + 1)

    return progress


def save_results(best_nnet, optimizer, results, best_result_index, args):
    dir = '../results/overview/'
    r_f, br_f = dir + 'results.csv', dir + 'best_result.csv'
    lvs_f, bm_f = dir + 'latent_vectors.csv', dir + 'best_model.pth.tar'
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    print(f'INFO: Finished. Saving experiment results to {r_f}.')
    results.to_csv(r_f)
    # print(f'INFO: Saving media figures with best index of {best_result_index}.')

    # best_result = results.loc[best_result_index]
    # best_result.to_csv(br_f, header=False)
    # args.batch_size, args.channels = best_result.batch_size, best_result.channels
    # full_train_loader, _ = load_dataset(args, distributed=False)

    # viz.visualize(best_nnet, full_train_loader, dir, args)
    save_checkpoint(args, {
        'state_dict': best_nnet.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }, filename=bm_f)
    # save_latent_vector(best_nnet, full_train_loader, lvs_f, args)


def save_latent_vector(nnet, train_loader, filename, args):
    print(f'INFO: Saving latent vector of model to {filename}.')
    # turn off gradients and other aspects of training
    nnet.eval()
    with torch.no_grad():
        n_batches  = len(train_loader)
        dictinary_list = []
        for i, (X,T,F) in enumerate(train_loader):
            X = X[:, args.channels, ...]
            if 'lin' in args.model:
                X = X.reshape(X.shape[0], np.product(X.shape[1:]))
            if args.cuda:
                X = X.cuda(non_blocking=True)
            if 'vae' in args.model:
                Y = nnet.encode(X)[0].detach().numpy()
            else:
                Y = nnet.encode(X).detach().numpy()

            for y, t, f in zip(Y, T, F):
                dictionary_data = {'latent_vector': list(y), 'label': t.item(), 'filename': f}
                dictinary_list.append(dictionary_data)
            printProgressBar(i + 1, n_batches, prefix='Progress:', suffix='Complete', length=50)

        latent_vectors = pd.DataFrame.from_dict(dictinary_list)
        latent_vectors.to_csv(filename)


def save_checkpoint(args, state, filename='best_model.pth.tar'):
    date = datetime.datetime.now().strftime('%m%d%Y')
    print(f'INFO: Saving best model to {filename} with lowest '
           'error from node rank: {0}'.format(dist.get_rank()))
    torch.save(state, filename)


def load_model(filename):
    r"""Currently NOT being used. This could be used for checkpointing,
    but needs additional logic to set the best_loss and start_epoch.
    """
    if os.path.isfile(filename):

        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filenamel)
        checkpoint.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return nnet, optimizer
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        sys.exit(3)


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1,
                      length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    if iteration == total:
        print()


if __name__ == '__main__':
    main()
