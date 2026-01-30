import torch.nn as nn
import numpy as np

import torch

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gru = torch.nn.GRU(128, 64, 1, batch_first=True, bidirectional=True)
                                
    def forward(self, inp):
        self.gru.flatten_parameters()
        out, _ = self.gru(inp)
        return out
                        

net = Net()
inp = torch.rand(32, 8, 128, requires_grad=True)

net = torch.nn.DataParallel(net)

inp = inp.cuda()
net = net.cuda()
out = net.forward(inp)