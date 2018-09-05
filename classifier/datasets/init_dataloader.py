from .init_dataset import ImageListDataset,FixedSubsetRandomSampler
import torch.utils.data
import os
import pandas as pd

def generate_loader(opt, split, inference_list = None):
    
    if split == 'train':
        current_transform = opt.train_transform
        current_shuffle = True
        #idxs = list(pd.read_csv(opt.data_list+'train_weights_idx_2.txt').coef.values)
        #sampler = FixedSubsetRandomSampler(idxs, opt.num_images_in_epoch)
        sampler = None
        
    else:
        current_transform = opt.test_transform
        current_shuffle = False
        sampler = None
        
  
    if inference_list:
        data_list = inference_list
        data_root= ''
        current_shuffle = False
    else:
        data_list = os.path.join(opt.data_list, split + '_list.txt')
        data_root = opt.data_root
        
    dataset = ImageListDataset(data_root = data_root,  data_list = data_list, transform=current_transform)
    assert dataset
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, shuffle = current_shuffle,
                                                 num_workers = int(opt.nthreads),sampler = sampler)
    return dataset_loader
