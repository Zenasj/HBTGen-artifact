import torch.nn as nn

from torch.autograd import Variable
import torch.onnx
import torchvision


x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
z = y.mean()

torch.onnx.export(z, x, 'cg.onnx.pb', verbose=True)

import torch
from torch.autograd import Variable
import torch.onnx

class Simple(torch.nn.Module):
    def forward(self, x):
        y = x * 2
        z = y.mean()
        return z


x = torch.randn(3)
x = Variable(x, requires_grad=True)
net = Simple()
torch.onnx.export(net, x, 'cg.onnx.pb', verbose=True)