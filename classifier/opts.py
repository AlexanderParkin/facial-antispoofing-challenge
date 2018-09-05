import argparse, os, json
import torch
import torchvision as tv
from utils import transforms

def get_opts():
    opt = argparse.Namespace()
    
    opt.task_name = ''
    opt.exp_name = ''
    
    opt.data_root = ''
    opt.data_list = ''
    
    opt.out_root = ''
    opt.out_path = os.path.join(opt.out_root,opt.task_name,opt.exp_name)
    
    ### Dataloader options ###
    opt.nthreads = 32
    opt.batch_size = 480
    opt.ngpu = 4

    ### Learning ###
    opt.lr = 0.001
    opt.lr_decay_lvl = 0.9
    opt.lr_decay_period = 15
    
    opt.num_epochs=200
    opt.num_images_in_epoch = 1000000
    opt.resume = ''
    opt.debug = 0
    ### Other ###  
    opt.manual_seed = 42
    opt.log_batch_interval=10
    opt.log_checkpoint = 4
    opt.net_type = 'mobilenetv2'
    opt.loss_type='bce'
    opt.loss_weights='1'
    opt.nclasses ='1'
    opt.hidden_layer_size = 0
    
    opt.train_transform = tv.transforms.Compose([
        tv.transforms.CenterCrop(224),
        #tv.transforms.Resize(120),
        tv.transforms.RandomResizedCrop(size=112, scale = (0.8,1.0), ratio=(0.9,1.1111)),
        transforms.GaussianBlur(5,0.25),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3),
        #tv.transforms.RandomCrop(size =(112,112)),
        
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
    ])

    opt.test_transform = tv.transforms.Compose([
        tv.transforms.CenterCrop(224),
        tv.transforms.Resize(112),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
    ])
    
    
    return opt


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--savepath', type=str, default = '/media/grinchuk/tmp/runners/', help = 'Path to save options')
    conf = parser.parse_args()
    opts = get_opts()
    filename = conf.savepath + opts.task_name + '_' + opts.exp_name+'.opt'
    torch.save(opts, filename)
    print('Options file was saved to '+filename)
