import torch
import torch.nn as nn

for momentum in [None, 1]:
     torch.manual_seed(32)
     bn = nn.BatchNorm1d(1, momentum=momentum)
     x = torch.rand(180,1,180)

     print(bn.running_var)
     bn.train()
     print(f'momentum = {momentum} yields {bn(x).mean()} for TRAIN mode')
     print(bn.running_var)

     bn.running_var.data.mul_(1 - 1 / (180*180))
     bn.eval()
     print(f'momentum = {momentum} yields {bn(x).mean()} for EVAL mode')