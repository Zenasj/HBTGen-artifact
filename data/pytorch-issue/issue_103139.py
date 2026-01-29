# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class Shake(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, inp1, inp2, training):
        assert inp1.size() == inp2.size()
        gate_size = [inp1.size()[0], *itertools.repeat(1, inp1.dim() - 1)]
        gate = inp1.new(*gate_size)
        if training:
            gate.uniform_(0, 1)
        else:
            gate.fill_(0.5)
        return inp1 * gate + inp2 * (1.0 - gate)

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_inp1 = grad_inp2 = grad_training = None
        gate_size = [grad_output.size()[0], *itertools.repeat(1, grad_output.dim() - 1)]
        gate = rad_output.data.new(*gate_size).uniform_(0, 1)  # Typo preserved as in original issue
        if ctx.needs_input_grad[0]:
            grad_inp1 = grad_output * gate
        if ctx.needs_input_grad[1]:
            grad_inp2 = grad_output * (1 - gate)
        assert not ctx.needs_input_grad[2]
        return grad_inp1, grad_inp2, grad_training

def shake(inp1, inp2, training=False):
    return Shake.apply(inp1, inp2, training)

class MyModel(nn.Module):
    @classmethod
    def out_channels(cls, planes, groups):
        assert groups == 1
        return planes

    def __init__(self, inplanes=4, planes=4, groups=1, stride=1, downsample=None):
        super().__init__()
        assert groups == 1
        self.conv_a1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv_b1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        a, b = x, x
        a = self.conv_a1(a)
        b = self.conv_b1(b)
        ab = shake(a, b, training=self.training)
        return ab

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 4, 4, 4, dtype=torch.float32)

