import argparse,json,random,os
import torch
import torch.backends.cudnn as cudnn
import torchvision as tv
import pandas as pd
from trainer import Model
from opts import get_opts
import datasets
from tqdm import tqdm

def extract_list():
    
    # Load options
    parser = argparse.ArgumentParser(description='Attribute Learner')
    parser.add_argument('--config', type = str, help = 'Path to config .opt file. Leave blank if loading from opts.py')
    parser.add_argument('--pth', type = str, help = 'Path to model checkpoint. Leave blank if testing bestmodel')
    parser.add_argument('--input_list', type = str, help = 'Path to list with image paths')
    parser.add_argument('--output_list', type = str, help = 'Path to list where to store results')
    conf = parser.parse_args()

    opt = torch.load(conf.config) if conf.config else get_opts()
    opt.ngpu = 1
    opt.batch_size=16
    print('Loading model ...')
    M = Model(opt)
    checkpoint = torch.load(conf.pth)
      
    try:
        checkpoint = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
    except:
        pass
    
    M.model.load_state_dict(checkpoint)
    M.model.eval()
    
    test_loader = datasets.generate_loader(opt, 'test', conf.input_list)
    
    torch.set_grad_enabled(False)
    out_f = open(conf.output_list,'w')
    
    for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
        #print('Extracting batch # {batch_idx} ...')
        data=data.to(M.device)
        output = M.model(data)
        output = torch.cat(output,1).detach().cpu().numpy()
        log_str='\n'.join(map(lambda x: ','.join(map(str,x)),output))+'\n'
        out_f.write(log_str)
    out_f.close()
    
    print('Extracting done!')
    
if __name__=='__main__':
    extract_list()
    