import torch
import torch.nn as nn
from torch.autograd import Variable

class ComputeFunction(nn.Module):
    def __init__(self):
        super(ComputeFunction, self).__init__()
        
    def forward(self, x):
        return x**2
    
func = ComputeFunction()

class ComputeGrads(nn.Module):
    def __init__(self):
        super(ComputeGrads, self).__init__()
        
    def forward(self, x):
        
        a = func(x)
        input_grads = torch.autograd.grad(outputs=torch.abs(a).sum(),
                    inputs=x, retain_graph=True)[0]
        return input_grads
    
func_grad = ComputeGrads()

dummy_input = Variable(torch.FloatTensor([2]), requires_grad=True)
func_grad = func_grad.eval()
torch.onnx.export(func_grad, dummy_input, "test.onnx", verbose=True)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np


writer = SummaryWriter()

from torch import nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        

    def forward(self, x):
        y = ((x*x)+2).sum()
        return torch.autograd.grad(y,x,retain_graph=True,create_graph=True)


net = Net()

net(torch.ones((1,3,3,2),requires_grad=True))
writer.add_graph(net, input_to_model=torch.ones((1,3,3,2),requires_grad=True), verbose=True)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchviz import make_dot, make_dot_from_trace

writer = SummaryWriter()

from torch import nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        

    def forward(self, x):
        y = ((torch.fft(x,2)*x)+2).sum()
        return torch.autograd.grad(y,x,retain_graph=True,create_graph=True)[0]


net = Net()
inp=torch.ones((1,3,3,2),requires_grad=True)
y=net(inp)
# writer.add_graph(net, input_to_model=torch.ones((1,3,3,2),requires_grad=True), verbose=True)
make_dot(y, params=dict(list(net.named_parameters())+[("input",inp)] ))