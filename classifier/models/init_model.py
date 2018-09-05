from .architectures import MobileNetV2

import torch
import torch.nn as nn

def init_body(arch_type,pretrained = False):
    """
    Available models:
    shufflenet_51
    shufflenet_52
    """
    if arch_type == 'mobilenetv2':
        model = MobileNetV2()
        if pretrained:
            pretrained_model = torch.load('/media/grinchuk/at_learner/models/pretrained/mobilenetv2.pth')
            model.load_state_dict(pretrained_model)
        return model
    else:
        raise Exception('Unknown architecture type')
        
def init_clf(input_size, output_size, nonlin = nn.Sigmoid(), hidden_layer_size=0):
    clf_modules=[]
    if hidden_layer_size:
        clf_modules.append(nn.Linear(input_size, hidden_layer_size))
        clf_modules.append(nn.ReLU(inplace=True))
        clf_modules.append(nn.Linear(hidden_layer_size, output_size))
    else:
        clf_modules.append(nn.Linear(input_size, output_size))
    if nonlin:
        clf_modules.append(nonlin)
    return nn.Sequential(*clf_modules)
    
def init_loss(criterion, reduction = 'elementwise_mean'):
    criterion_names = criterion.split('_')
    criterion_list = list()
    for criterion_name in criterion_names:
        if criterion_name=='bce':
            loss = nn.BCELoss(reduction=reduction)
        elif criterion_name=='mse':
            loss = nn.MSELoss(reduction=reduction)
        elif criterion_name=='cce':
            loss = nn.CrossEntropyLoss(reduction=reduction)
        else:
            raise Exception('This loss function is not implemented yet.')
        criterion_list.append(loss)
                     
    return criterion_list
    
    
    
    
