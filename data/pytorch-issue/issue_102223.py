import torch
from torch.utils.checkpoint import checkpoint

def run_fn(tensor):
    y = tensor * 2
    z = y.sum()
    grads = torch.autograd.grad(z, tensor)
    return grads[0]

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = checkpoint(run_fn, x, use_reentrant=False)
z = y.sum()
z.backward()
print(x.grad)

import torch
from torch.utils.checkpoint import checkpoint

def run_fn(tensor):
    y = tensor.exp()
    z = y.sum()
    grads = torch.autograd.grad(z, tensor, create_graph=True)
    return grads[0]

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = checkpoint(run_fn, x, use_reentrant=False)
z = y.sum()
z.backward()
print(x.grad)

import torch
from torch.utils.checkpoint import checkpoint

def run_fn(x):
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        y = torch.exp(x)
        z = y.sum()
        grads = torch.autograd.grad(z, x, create_graph=create_graph)[0]
    return grads if grads.requires_grad else grads.clone().requires_grad_(True)

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = checkpoint(run_fn, x)
z = y.sum()
z.backward()
print(x.grad)

import torch
from torch.utils.checkpoint import checkpoint

def run_fn(tensor):
    print('Running with use_reentrant=False and autograd.grad')
    y = tensor.exp()
    z = y.sum()
    grads = torch.autograd.grad(z, tensor, create_graph=True)
    return grads[0]

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = checkpoint(run_fn, x, use_reentrant=False)
z = y.sum()
z.backward()
print('Grad with use_reentrant=False', x.grad)

import torch
from torch.utils.checkpoint import checkpoint

def run_fn(x):
    print('Running with use_reentrant=True and autograd.grad')
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        y = torch.exp(x)
        z = y.sum()
        grads = torch.autograd.grad(z, x, create_graph=create_graph)[0]
    return grads

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = checkpoint(run_fn, x, use_reentrant=True)
z = y.sum()
z.backward()
print('Grad with use_reentrant=True + workaround', x.grad)

def run_fn(tensor):
    print('Running with use_reentrant=False and no autograd.grad')
    y = tensor.exp()
    z = y.sum()
    output = z + tensor
    return output

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = checkpoint(run_fn, x, use_reentrant=False)
z = y.sum()
z.backward()