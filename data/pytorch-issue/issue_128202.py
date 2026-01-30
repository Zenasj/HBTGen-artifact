import torch
import habana_frameworks.torch.core as htcore

def fn(a):
 b = a.t()
 b.mul_(1.0)
 return b   

x = torch.arange(6).reshape([2, 3]).to('hpu')

print("x ", x.cpu())

compiled_fn = torch.compile(fn, backend="hpu_backend")        
y = compiled_fn(x)

print("y ", y.cpu())

import torch

def fn(a):
 b = a.t()
 b.mul_(1.0)
 return b

x = torch.arange(6).reshape([2, 3]).to('cpu')

print("x ", x.cpu())

compiled_fn = torch.compile(fn)
y = compiled_fn(x)

print("y ", y.cpu())

import torch

def fn(a):
 b = a.t()
 b.mul_(1.0)
 return b

x = torch.arange(6).reshape([2, 3]).bfloat16()

print("x ", x.cpu())

compiled_fn = torch.compile(fn)
y = compiled_fn(x)

print("y ", y.cpu())