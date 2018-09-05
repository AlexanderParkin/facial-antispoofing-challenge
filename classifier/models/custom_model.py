from .init_model import init_body, init_clf

import torch
import torch.nn as nn
        
class IDRND_Liveness_v1(nn.Module):
    def __init__(self, body_arch, hidden_layer_size = 0):
        super(IDRND_Liveness_v1, self).__init__()
        self.body = init_body(body_arch)
        self.descriptor_size = 1024 if ((body_arch=='shufflenet_51')) else 256

        self.ir_clf = init_clf(self.descriptor_size, 1, nn.Sigmoid(), hidden_layer_size)

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), -1)
        x_ir = self.ir_clf(x)
        return [x_ir]       
        
