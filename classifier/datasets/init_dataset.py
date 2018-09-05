import torch.utils.data as data
import torch
from PIL import Image
import pandas as pd
import numpy as np
import os

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def fast_loader(path):
    return Image.open(path)        
        
def bw_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')        


class ImageListDataset(data.Dataset):
    """
    Builds a dataset based on a list of images.
    data_root - image path prefix
    data_list - annotation list location
    """
    def __init__(self, data_root, data_list, transform=None, target_transform=None, im_type = 'RGB'):
        self.data_root = data_root
        self.data_list = data_list
        self.images, self.targets = self.read_list()
        self.transform = transform
        self.target_transform = target_transform
        self.loader = fast_loader if im_type == 'RGB' else bw_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) 
        """
        path = self.images[index]
        target = self.targets[index]
        
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)
    
    def read_list(self):
        data_df = pd.read_csv(self.data_list)
        images = data_df.values[:,0]
        targets = np.float32(data_df.values[:,1:])
        return self.data_root+images,targets
    
    
    
class FixedSubsetRandomSampler(data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices, epoch_size = None):
        self.indices = indices
        self.epoch_size = epoch_size if epoch_size else len(self.indices)
    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices))[:self.epoch_size])

    def __len__(self):
        return self.epoch_size    
  