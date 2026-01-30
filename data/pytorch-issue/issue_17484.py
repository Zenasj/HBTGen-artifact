import torch
import torch.nn as nn

class A(torch.nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.par = torch.nn.Parameter(torch.rand(5)).to('cuda')
    def forward(self):
        pass
    
a= A()
a.state_dict()
Out[8]: OrderedDict()

self.par = torch.nn.Parameter(torch.rand(5, device='cuda'))