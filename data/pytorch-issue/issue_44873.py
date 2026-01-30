import torch.nn as nn

import torch
from torch.nn import DataParallel

class A(torch.nn.Module):
    def forward(self):
        pass

a = A() 
a = DataParallel(a, device_ids=[0, 1]) 
a()