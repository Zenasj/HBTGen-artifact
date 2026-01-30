import torch
import torch.nn as nn

class net(torch.nn.Module):
    def __init__(self):
        super(net,self).__init__()
    def forward(self,x):
        return x.bool()
model = net()
ts = torch.jit.script(model)