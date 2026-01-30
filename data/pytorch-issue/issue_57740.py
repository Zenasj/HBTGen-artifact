import torch.nn as nn

import torch

class Net(torch.nn.Module):
    def forward(self):
        self.register_buffer('test', torch.ones(1))
        return self.test

net = Net()
net() #== tensor([1.]) 
torch.jit.script(net) # RuntimeError