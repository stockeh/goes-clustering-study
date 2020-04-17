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

def main():
    parser = argparse.ArgumentParser(description="Distributed PyTorch")

    parser.add_argument('world_size', type=int, default=4, metavar='N',
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""")

    parser.add_argument('rank', type=int, default=None, metavar='N',
        help="Global rank of this process. Pass in 0 for master.")

    parser.add_argument('--dist-conn', type=str, default='bentley:15516',
                        help='master host:port used to set up distributed training')

    parser.add_argument('--batch-size', type=int, default=25, metavar='N',
        help='mini-batch size (default: 3).')

    parser.add_argument('--epochs', type=int, default=1, metavar='N',
        help='number of total epochs to run')

    parser.add_argument("--data", type=str, default="../data/", metavar='N',
        help="Directory containing the data to be run on.")

    parser.add_argument('--cuda', type=bool, default=False, metavar='N',
        help='True to use CUDA on GPU. WARN: only implemented on CPU!.')

    parser.add_argument('--latent-dim', type=int, default=10, metavar='N',
        help='Dimensionality of the latent vector.')

    parser.add_argument('--channels', type=int, nargs='+', default=[0, 6], metavar='N',
        help='Channels to evaluate (use at end).')

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
    assert len(args.channels) > 0, f'{args.channels} :: must provide valid channels, e.g., [0, 2, 3].'
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


def main_worker(args):
    global best_loss, best_nnet

    train_loader, train_sampler = load_dataset(args)

    results = pd.DataFrame(columns=['algo', 'rho', 'epochs', 'batch_size',
                                    'latent_dim', 'hidden_dims', 'ker_str_pad',
                                    'error_trace', 'duration'])
    algo    = 'adam'
    l_epoch = [2]
    l_rho   = [0.01]
    l_latent_dim = [2]
    l_hidden_dims = [[4, 8, 16]]
    l_ker_str_pad = [[(10, 3, 1), (8, 2, 0), (3, 3, 0)]]

    for (epochs, rho, latent_dim, hidden_dims, ker_str_pad) in itertools.product(
        l_epoch, l_rho, l_latent_dim, l_hidden_dims, l_ker_str_pad):

        args.hidden_dims, args.ker_str_pad, args.latent_dim = hidden_dims, ker_str_pad, latent_dim
        nnet = models.get_model(train_loader, args)

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
        error_trace = []

        for epoch in range(epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            # train for one epoch
            progress = train(train_loader, nnet, criterion,
                             optimizer, epoch, error_trace, args)

        results.loc[len(results)] = [algo, rho, epochs, args.batch_size,
                                     latent_dim, hidden_dims, ker_str_pad,
                                     error_trace, time.time() - start]

        loss = progress.getAvgMeter('Error').avg

        # remember best error and save nnet
        if loss < best_loss:
            best_loss = loss
            best_nnet = nnet

    if dist.get_rank() == 0:
        save_results(best_nnet, optimizer, train_loader, results, args)


def train(train_loader, nnet, criterion, optimizer, epoch, error_trace, args):
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

        error_trace.append(loss.item())
        error.update(loss.item(), X.shape[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # display progress
        progress.display(i + 1)

    return progress


def save_results(best_nnet, optimizer, train_loader, results, args):
    r_f, lvs_f, bm_f = 'results.csv', 'latent_vectors.csv', 'best_model.pth.tar'
    print(f'INFO: Finished. Saving experiment results to {r_f}.')
    results.to_csv(r_f)
    print('INFO: Saving media figures.')
    if args.distributed:
        best_nnet.require_forward_param_sync = False
    viz.visualize(best_nnet, train_loader, args)
    # save_checkpoint(args, {
    #     'state_dict': best_nnet.state_dict(),
    #     'optimizer' : optimizer.state_dict(),
    # }, filename=bm_f)
    print(f'INFO: Saving latent vector of model to {lvs_f}.')
    latent_vectors = pd.DataFrame(columns=['latent_vector'])
    for i, X in enumerate(train_loader):
        X = X[:, args.channels, ...]
        if 'lin' in args.model:
            X = X.reshape(X.shape[0], np.product(X.shape[1:]))
        if args.cuda:
            X = X.cuda(non_blocking=True)
        Y = best_nnet.encode(X).detach().numpy()
        for batch in Y:
            latent_vectors.loc[len(latent_vectors)] = [list(batch)]
    latent_vectors.to_csv(lvs_f)

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


if __name__ == '__main__':
    main()
