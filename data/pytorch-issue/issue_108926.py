import torch.nn as nn

import torch

def f(x):
    return torch.sum(x[:1])

class EvaluateGradient(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        y = self.f(x)
        grads = torch.autograd.grad(y, x, create_graph=True)
        return grads[0]

x = torch.zeros((2, 2))
x.requires_grad_()

print(f(x))

grad = EvaluateGradient(f)

print(grad(x))

torch.onnx.export(grad, x, "model.onnx")

import torch

def f(x):
    return torch.sum(x[:1])

class EvaluateGradient(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        y = self.f(x)
        grads = torch.autograd.grad(y, x, create_graph=True)
        return grads[0]

x = torch.zeros((2, 2))
x.requires_grad_()

print(f(x))

grad = EvaluateGradient(f)
torch.export.export(grad, args=(x,))

import torch

def f(x):
    return torch.sum(x[:1])

class EvaluateGradient(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        y = self.f(x)
        grads = torch.autograd.grad(y, x, create_graph=True)
        return grads[0]

x = torch.zeros((2, 2))
x.requires_grad_()

print(f(x))

grad = EvaluateGradient(f)
torch.onnx.dynamo_export(grad, x)