import torch


def extract(V):
    def hook(grad):
        V.grad = grad
    return hook

X = torch.ones(5, requires_grad=True)
V = X.pow(3)
Y = V.sum()

V.register_hook(extract(V))
Y.backward()

print('X.grad:', X.grad)
print('V.grad:', V.grad)

import torch


def extract_and_grad(V):
    def hook(grad):
        V.grad = grad
        V.grad.requires_grad_(True)
        print('V.grad:', V.grad)
        U = V.grad.pow(2).sum()
        print('U:', U)
        try:
            dU_dV_grad = torch.autograd.grad(U, V.grad)
            print('V.grad.grad:', dU_dV_grad)
        except RuntimeError as err:
            print(err)
    return hook

X = torch.ones(5, requires_grad=True)
V = X.pow(3)
Y = V.sum()

V.register_hook(extract_and_grad(V))
Y.backward()

import torch


def extract_and_grad_v2(V):
    def hook(grad):
        V.grad = grad
        V.grad.requires_grad_(True)
        print('V.grad:', V.grad)
        
        with torch.enable_grad():
            U = V.grad.pow(2).sum()
            print('U:', U)
        try:
            dU_dV_grad, = torch.autograd.grad(U, V.grad)
            print('V.grad.grad:', dU_dV_grad)
        except RuntimeError as err:
            print(err)
    return hook

X = torch.ones(5, requires_grad=True)
V = X.pow(3)
Y = V.sum()

V.register_hook(extract_and_grad_v2(V))
Y.backward()

import torch


def extract(V):
    def hook(grad):
        V.grad = grad
    return hook

X = torch.ones(5, requires_grad=True)
V = X.pow(3)
Y = V.sum()

V.register_hook(extract(V))
Y.backward()

V.grad.requires_grad_(True)
print('X.grad:', X.grad)
print('V.grad:', V.grad)
U = V.grad.pow(2).sum()
print('U:', U)
try:
    dU_dV_grad, = torch.autograd.grad(U, V.grad)
    print('V.grad.grad:', dU_dV_grad)
except RuntimeError as err:
    print(err)