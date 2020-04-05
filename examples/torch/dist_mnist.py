import os
import sys
import traceback
import datetime
import socket
import random
import subprocess
import math
import time
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.multiprocessing import Process
from torchvision import datasets, transforms
from skimage.transform import resize

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

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = random.Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def partition_dataset():
    # root_data = 'hdfs://olympia:32351/'
    root_data = './data'
    dataset = datasets.MNIST(root_data, train=True, download=True,
                             transform = transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                              ]))
    size = dist.get_world_size()
    bsz = int(128 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes).use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True)

    return train_set, bsz

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    if iteration == total:
        print()

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def run(rank, world_size):

    Xtrain, bsz = partition_dataset()
    CUDA = False
    if torch.cuda.is_available():
        pass # CUDA = True

    if CUDA:
        nnet = nn.parallel.DistributedDataParallel(Net()).float().cuda()
        print('Running on CUDA')
    else:
        nnet = nn.parallel.DistributedDataParallel(Net()).float()

    optimizer = optim.SGD(nnet.parameters(), lr=0.01, momentum=0.50)
    criterion = nn.CrossEntropyLoss()

    num_batches = np.ceil(len(Xtrain.dataset) / float(bsz))
    best_loss = float('inf')
    for epoch in range(3):
        epoch_loss = 0.0
        printProgressBar(0, len(Xtrain), prefix='Progress:', suffix='Complete', length=50)

        for i, (X, T) in enumerate(Xtrain):
            if CUDA:
                X, T = X.cuda(), T.cuda()

            optimizer.zero_grad()

            Y = nnet(X)
            loss = criterion(Y, T)
            epoch_loss += loss.item()
            loss.backward()
            
            average_gradients(nnet)
            optimizer.step()
            printProgressBar(i + 1, len(Xtrain), prefix='Progress:', suffix='Complete', length=50)

        print(f'Rank {dist.get_rank()}, epoch {epoch}: {epoch_loss / num_batches}')
        if dist.get_rank() == 0 and epoch_loss / num_batches < best_loss:
            best_loss = epoch_loss / num_batches
            torch.save(nnet.state_dict(), 'best_model.pth')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'mars'
    os.environ['MASTER_PORT'] = '15515'

    # init process group
    dist.init_process_group("gloo", rank=int(rank), world_size=int(world_size),
            init_method='tcp://mars:15516', timeout=datetime.timedelta(weeks=120))

    # explicitly set seed for all models to start with same w
    torch.manual_seed(42)

if __name__ == '__main__':
    try:
        rank, world_size = int(sys.argv[1]), int(sys.argv[2])
        setup(rank, world_size)
        print(socket.gethostname() + ': setup completed!')
        start = time.time()
        run(rank, world_size)
        print (f'total: {time.time() - start} (s)')
    except Exception as e:
        traceback.print_exc()
        sys.exit(3)
