import glob
import torch
import numpy as np

from torch.utils.data import Dataset

import read_region as rr


class GOESDataset(Dataset):
    """Eastern Pacific/Northern Atlantic Ocean GOES NetCDF dataset."""

    def __init__(self, root_dir, transform=None ):
        """
        Args:
            root_dir (string): directory with all the .nc files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir  = root_dir
        self.data_list = glob.glob(root_dir + '*')
        self.data_len  = len(self.data_list)

        self.channels   = True
        self.geo        = False
        self.retrievals = True

        self.transform = transform

    def __getitem__(self, inx):
        """WARN: Currently only implemented for `channels` and `retrivals`."""
        try:
            filename = self.data_list[inx]
            data = rr.read_region(filename, channels=self.channels,
                                   geo=self.geo, retrievals=self.retrievals)

            sample, type = np.array(data['c00']), np.array(data['ctype'])
            sample = np.moveaxis(sample, 2, 0)

            counts = np.bincount(type.flatten())
            label = np.argmax(counts)

            if self.transform:
                sample = self.transform(sample)

            return sample, label, filename
        except:
            # used too catch invalid samples, and custom_collate
            pass

    def __len__(self):
        return self.data_len

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root_dir is not None:
            body.append("Root location: {}".format(self.root_dir))
        body += "".splitlines()
        if self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [line for line in body]
        return '\n'.join(lines)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
       Scale data with the max values from each channel
    """
    def __init__(self):

        self.max_channels = np.array([  1.23364961,   1.1933322 ,   1.20698297,   0.77238023,
                                        0.74793577,   0.57047564, 313.15881348, 256.77209473,
                                        264.97137451, 272.53656006, 294.66796875, 276.41287231,
                                        297.76239014, 297.10894775, 294.52630615, 279.70178223]).reshape(16, 1, 1)

    def __call__(self, sample):
        # torch shape: C X H X W
        sample = sample / self.max_channels
        return torch.from_numpy(sample).float()


    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize individual sample with its own min/max values."""

    def __call__(self, sample):
        # torch shape: C X H X W
        Xmin = np.min(sample, axis=tuple([1, 2])).reshape(sample.shape[0], 1, 1)
        Xmax = np.max(sample, axis=tuple([1, 2])).reshape(sample.shape[0], 1, 1)
        diff = (Xmax - Xmin)
        if diff.all() == 0:
            return None

        return (sample - Xmin)/diff

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Square(object):
    """Convert the sample to a square based on min dimension.
    This should be done prior to Noramlize.
    """

    def __call__(self, sample):
        # torch shape: C X H X W
        nnew = min(sample.shape[1], sample.shape[2])
        return sample[:, 0:nnew, 0:nnew]

    def __repr__(self):
        return self.__class__.__name__ + '()'


def custom_collate(batch):
    # remove items
    batch = list(filter(lambda x : x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
