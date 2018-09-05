import argparse, os, json
import torch
import torchvision as tv
from utils import transforms
import PIL

def get_opts():
    opt = argparse.Namespace()
    
    opt.task_name = 'IDRND_Spoofing'
    opt.exp_name = 'baseline_mobilenet'
    
    opt.data_root = '/media/a.parkin/media/CameraSpoofing/'
    opt.data_list = '/media/a.parkin/media/CameraSpoofing/lists/baseline/'
    
    opt.out_root = '/media/a.parkin/media/CameraSpoofing/results/'
    opt.out_path = os.path.join(opt.out_root,opt.task_name,opt.exp_name)
    
    ### Dataloader options ###
    opt.nthreads = 32
    opt.batch_size = 256
    opt.ngpu = 1

    ### Learning ###
    opt.lr = 0.0002
    opt.lr_decay_lvl = 0.9
    opt.lr_decay_period = 50
    
    opt.num_epochs=100
    opt.num_images_in_epoch = 1000
    opt.resume = ''
    opt.debug = 0
    ### Other ###  
    opt.manual_seed = 42
    opt.log_batch_interval=10
    opt.log_checkpoint = 4
    opt.net_type = 'mobilenetv2'
    opt.loss_types='bce'
    opt.loss_weights='1'
    opt.nclasses ='1'
    opt.hidden_layer_size = 0
    
    opt.train_transform = tv.transforms.Compose([
        tv.transforms.RandomRotation(30,resample=PIL.Image.BICUBIC,expand=False),
        transforms.CenterRandomSizeCrop(0.4,0.8),
        tv.transforms.RandomChoice([
            tv.transforms.Compose([
                tv.transforms.RandomChoice([
                    transforms.RandomResizeBySize(0.75),
                    transforms.RandomCropBySize(0.75),
                ]),
                tv.transforms.Resize(168, interpolation=3)
            ]),
            tv.transforms.Compose([
                tv.transforms.Resize(224, interpolation=3),
                tv.transforms.RandomChoice([
                    tv.transforms.RandomResizedCrop(168, interpolation=3),
                    tv.transforms.RandomCrop(168),
                ])
            ]),
        ]),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3),
        transforms.GaussianBlur(5,0.25),
        tv.transforms.Resize(112, interpolation=3),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
    ])

    opt.test_transform = tv.transforms.Compose([
        transforms.CenterCropBySize(0.6),
        tv.transforms.Resize(224, interpolation=3),
        tv.transforms.CenterCrop(168),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.Resize(112, interpolation=3),
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
