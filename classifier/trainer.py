import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time, os

import models, datasets, utils

class Model:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if opt.ngpu else "cpu")
        
        self.model = models.AgeNet_v4(opt.net_type,opt.hidden_layer_size).to(self.device)
        #self.model = models.IR_Liveness_v1(opt.net_type, opt.hidden_layer_size).to(self.device)
        if opt.ngpu>1:
            self.model = nn.DataParallel(self.model)
            
        self.loss = models.init_loss(opt.loss_types)
        self.loss = list(map(lambda x: x.to(self.device),self.loss))
        
        self.optimizer = optim.Adam(self.model.parameters(), lr = opt.lr, betas=(0.5, 0.999))
        
        self.train_loader = datasets.generate_loader(opt,'train') 
        self.test_loader = datasets.generate_loader(opt,'val')  
        
        self.nclasses = np.int32(opt.nclasses.split('_'))         #Class count list (e.g. 2_1 -> [2,1])
        self.loss_weights = torch.tensor(np.float32(opt.loss_weights.split('_'))).to(self.device) 
        self.loss_names = opt.loss_types.split('_')           #Criterion names list (e.g. bce_mse -> ['bce','mse']
    
        
        self.epoch = 0
        self.best_epoch = False
        self.training = False
        self.state = {}
        
        self.train_loss = utils.AverageMeter()
        self.test_loss  = utils.AverageMeter()
        self.batch_time = utils.AverageMeter()
        self.test_metrics = utils.AverageMeter()
        self.best_test_loss = utils.AverageMeter()                       
        self.best_test_loss.update(np.array([np.inf]*(len(self.nclasses))))
        

    def train(self):
        
        # Init Log file
       
        if self.opt.resume:
            self.log_msg('resuming...\n')
            # Continue training from checkpoint
            self.load_checkpoint()
        else:
             self.log_msg()
        for epoch in range(self.epoch, self.opt.num_epochs):
            self.epoch = epoch
            self.adjust_lr()
            self.train_epoch()
            self.test_epoch()
            self.log_epoch()
            self.create_state()
            self.save_state()
    
    
    
    def calc_loss(self, output, target, loss_idx):
        mask = target>=0
        loss_coef = mask.float().mean().item()
        if loss_coef==1:
            return self.loss[loss_idx](output,target),loss_coef
        return self.loss[loss_idx](output[mask],target[mask]),loss_coef
    
    
    def train_epoch(self):
        """
        Trains model for 1 epoch
        """
        self.model.train()
        self.training = True
        torch.set_grad_enabled(self.training)
        self.train_loss.reset()
        self.batch_time.reset()
        time_stamp = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            
            data, target = data.to(self.device),target.to(self.device)
            target = list(chunks(target, self.nclasses)) 
            self.optimizer.zero_grad()
            output = self.model(data)
            
            loss_tensor = torch.tensor(0.).to(self.device)
            loss_values = []
            loss_coeffs = []
            for loss_idx in range(len(self.nclasses)):
                current_target = target[loss_idx].squeeze(1).long() if self.loss_names[loss_idx]=='cce' else target[loss_idx]
                #current_loss,current_loss_coef = self.loss[loss_idx](output[loss_idx], current_target),1
                current_loss, current_loss_coef = self.calc_loss(output[loss_idx], current_target, loss_idx)
                loss_tensor += current_loss * self.loss_weights[loss_idx]
                loss_values.append(current_loss.item())
                loss_coeffs.append(current_loss_coef)
            loss_tensor.backward()            
            self.optimizer.step()
            self.train_loss.update(np.array(loss_values),np.array(loss_coeffs))
            
            self.batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()
            
            self.log_batch(batch_idx)
            if self.opt.debug and (batch_idx==10):
                print('Debugging done!')
                break;
            
    def test_epoch(self):
        """
        Calculates loss and metrics for test set
        """
        self.training = False
        torch.set_grad_enabled(self.training)
        self.model.eval()
        
        self.batch_time.reset()
        self.test_loss.reset()
        self.test_metrics.reset()
        time_stamp = time.time()
        
        for batch_idx, (data, target) in enumerate(self.test_loader):
            
            data, target = data.to(self.device),target.to(self.device)
            target = list(chunks(target, self.nclasses))
            output = self.model(data)
            #try:
            loss_values = []
            loss_coeffs = []
            metrics=[]
            for loss_idx in range(len(self.nclasses)):
                current_target = target[loss_idx].squeeze(1).long() if self.loss_names[loss_idx]=='cce' else target[loss_idx]
                #current_loss = self.loss[loss_idx](output[loss_idx], current_target)
                try:
                    current_loss, current_loss_coef = self.calc_loss(output[loss_idx], current_target, loss_idx)
                    loss_values.append(current_loss.item())
                    loss_coeffs.append(current_loss_coef)
                    current_metrics = self.calculate_metrics(output[loss_idx], current_target, loss_idx)
                    metrics.append(current_metrics)
                except:
                    loss_values.append(0)
                    loss_coeffs.append(0)
                    metrics.append([0])
                    print('Broken batch')
            #return metrics
            self.test_loss.update(np.array(loss_values),np.array(loss_coeffs))
            self.test_metrics.update(np.hstack(np.ravel(metrics)),np.array(loss_coeffs))
            #except:
            #    print('Skipped batch')
            self.batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()
            
            self.log_batch(batch_idx)
            
        
        
            if self.opt.debug and (batch_idx==10):
                print('Debugging done!')
                break;
        self.best_epoch = sum(self.test_loss.avg) < sum(self.best_test_loss.val)
        if self.best_epoch:
            # self.best_test_loss.val is container for best loss, 
            # n is not used in the calculation
            self.best_test_loss.update(self.test_loss.avg, n=0)
     
    def calculate_metrics(self, output, target, loss_idx):   
        """
        Calculates test metrix for given batch and its input
        """
        loss_name = self.loss_names[loss_idx]
        batch_result=[]
        
        
        mask = target>=0
        loss_coef = mask.float().mean().item()
        if loss_coef<1:
            t = target[mask]
            o = output[mask]
        else:
            t = target
            o = output
            
        if loss_name == 'bce':
            binary_accuracy = (t.byte()==(o>0.5)).float().mean(0).cpu().numpy()  
            batch_result.append(binary_accuracy)
        elif loss_name =='mse':
            mean_average_error = torch.abs(t-o).mean(0).cpu().numpy()
            batch_result.append(mean_average_error)
        elif loss_name == 'cce':
            top1_accuracy = (torch.argmax(o, 1)==t).float().mean().item()
            batch_result.append(top1_accuracy)
        else:
            raise Exception('This loss function is not implemented yet')
                
        return batch_result            
    
    def log_batch(self, batch_idx):
        if batch_idx % self.opt.log_batch_interval == 0:
            cur_len = len(self.train_loader) if self.training else len(self.test_loader)
            cur_loss = self.train_loss if self.training else self.test_loss
            
            output_string = 'Train ' if self.training else 'Test '
            output_string +='Epoch {}[{:.2f}%]: [{:.2f}({:.3f}) s]\t'.format(self.epoch,
                                                                          100.* batch_idx/cur_len, self.batch_time.val,self.batch_time.avg)
            for loss_idx in range(len(self.nclasses)):
                loss_i_string = 'Loss_{}: {:.3f}({:.3f})\t'.format(loss_idx, cur_loss.val[loss_idx], cur_loss.avg[loss_idx])
                output_string += loss_i_string
                    
            if not self.training:
                output_string+='\n'
                for loss_idx in range(sum(self.nclasses)):
                    metrics_i_string = 'Class_{}: {:.3f}({:.3f})\t'.format(loss_idx, self.test_metrics.val[loss_idx],
                                                                           self.test_metrics.avg[loss_idx])
                    output_string += metrics_i_string
                
            print(output_string)
    
    
    def log_msg(self, msg=''):
        mode = 'a' if msg else 'w'
        f = open(os.path.join(self.opt.out_path, 'log_files', 'train_log.txt'), mode)
        f.write(msg)
        f.close()
             
    def log_epoch(self):
        """ Epoch results log string"""
        out_train = 'Train: '
        out_test = 'Test:  '
        for idx in range(len(self.nclasses)):
            loss_i_string = 'Loss_{}: {:.3f}\t'.format(idx,self.train_loss.avg[idx])
            out_train += loss_i_string
            loss_i_string = 'Loss_{}: {:.3f}\t'.format(idx,self.test_loss.avg[idx])
            out_test += loss_i_string
            
        out_test+='\nTest:  '
        for idx in range(sum(self.nclasses)):
            metrics_i_string = 'Class_{}: {:.3f}\t'.format(idx,self.test_metrics.avg[idx])
            out_test += metrics_i_string
            
        is_best = 'Best ' if self.best_epoch else ''
        out_res = is_best+'Epoch {} results:\n'.format(self.epoch)+out_train+'\n'+out_test+'\n'
        
        print(out_res)
        self.log_msg(out_res)
        
    def adjust_lr(self):
        """Set the LR to the initial LR decayed by lr_decay_lvl every lr_decay_period epochs"""
        lr = self.opt.lr * (self.opt.lr_decay_lvl ** ((self.epoch+1) / self.opt.lr_decay_period))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr  
                    
    def create_state(self):
        self.state = {       # Params to be saved in checkpoint
                'epoch' : self.epoch,
                'state_dict' : self.model.state_dict(),
                'best_test_loss' : self.best_test_loss,
                'optimizer': self.optimizer.state_dict()
            }
    
    def save_state(self):
        if self.opt.log_checkpoint == 0:
                self.save_checkpoint('checkpoint.pth')
        else:
            if (self.epoch % self.opt.log_checkpoint == 0):
                self.save_checkpoint('model_{}.pth'.format(self.epoch)) 
                  
    def save_checkpoint(self, filename):     # Save model to task_name/checkpoints/filename.pth
        fin_path = os.path.join(self.opt.out_path,'checkpoints', filename)
        torch.save(self.state, fin_path)
        if self.best_epoch:
            best_fin_path = os.path.join(self.opt.out_path, 'checkpoints', 'model_best.pth')
            torch.save(self.state['state_dict'], best_fin_path)
           
                   
    def load_checkpoint(self):                            # Load current checkpoint if exists
        fin_path = os.path.join(self.opt.out_path,'checkpoints',self.opt.resume)
        if os.path.isfile(fin_path):
            print("=> loading checkpoint '{}'".format(fin_path))
            checkpoint = torch.load(fin_path, map_location=lambda storage, loc: storage)
            self.epoch = checkpoint['epoch'] + 1
            self.best_test_loss = checkpoint['best_test_loss']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})".format(self.opt.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(self.opt.resume))
            
            
def chunks(l, div):
    st=0
    for i in range(len(div)):
        yield l[:,st:st+div[i]]
        st+=div[i]            
            
            
