import glob
import torch
import numpy as np

from torch.utils.data import Dataset

import read_region as rr


class GOESDataset(Dataset):
    """Eastern Pacific/Northern Atlantic Ocean GOES NetCDF dataset."""
    
    def __init__(self, root_dir, channels=True, geo=False, 
                 retrievals=False, transform=None ):
        """
        Args:
            root_dir (string): directory with all the .nc files.
            channels (bool): set to True to return ABI channel data
            geo (bool): set to True to return geolocation data
            retrievals (bool): set to True to return retrieval data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir  = root_dir
        self.data_list = glob.glob(root_dir + '*')
        self.data_len  = len(self.data_list)
        
        self.channels   = channels
        self.geo        = geo
        self.retrievals = retrievals
        
        self.transform = transform
        
    def __getitem__(self, inx):
        """Retrive a single sample defined by the index.
        WARN: Currently only implemented for `channels`.
        Args:
            inx (int): index of file to retreive from glob.
        """
        filename = self.data_list[inx]
        
        data = rr.read_region(filename, channels=self.channels,
                               geo=self.geo, retrievals=self.retrievals)

        sample = np.array(data['c00'])
        sample = np.moveaxis(sample, 2, 0)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
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
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        # torch shape: C X H X W
        return torch.from_numpy(sample).float()
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
class Normalize(object):
    """Normalize sample with its own min/max values."""
    
    def __call__(self, sample):
        # torch shape: C X H X W
        Xmin = np.min(sample, axis=tuple([1, 2])).reshape(sample.shape[0], 1, 1)
        Xmax = np.max(sample, axis=tuple([1, 2])).reshape(sample.shape[0], 1, 1)
        return (sample - Xmin)/(Xmax - Xmin)
    
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