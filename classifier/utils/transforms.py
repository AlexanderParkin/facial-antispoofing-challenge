from __future__ import division
import torchvision as tv
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance,ImageFilter
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings
import cv2

from torchvision.transforms import functional as F

class GaussianBlur(object):
    """ Apply Gaussian blur to image with probability 0.5
    """
    def __init__(self,max_blur_kernel_radius=3,rand_prob=0.5):
        self.max_radius = max_blur_kernel_radius
        self.rand_prob = rand_prob
    def __call__(self, img):
        radius = random.uniform(0,self.max_radius)
        if random.random()<self.rand_prob:
            return img.filter(ImageFilter.GaussianBlur(radius))
        else:
            return img
    
    def __repr__(self):
        return self.__class__.__name__ + '({0})'.format(self.max_radius)

class ResizeOpencv(object):
    '''Apply resize with opencv function'''
    def __init__(self, size, interpolation=cv2.INTER_LINEAR, out_type='PIL'):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        if type(size) == int:
            self.size = (size, size)
        else:
            self.size = size
        self.interpolation = interpolation
        self.out_type = out_type
    
    def __call__(self, img):
        if type(img) != np.ndarray:
            img = np.array(img)
        img = cv2.resize(img, self.size, interpolation=self.interpolation)
        if self.out_type == 'PIL':
           img = Image.fromarray(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0},interpolation={1},out_type={2})'.format(self.size, self.interpolation, self.out_type)

class MeanStdNormalize(object):
    ''' Mean/Std normalize for torch tensor'''
    def __call__(self, tensor):
        return (tensor - tensor.mean())/tensor.std()
   
    def __repr__(self):
        return self.__class__.__name__ + '()'

