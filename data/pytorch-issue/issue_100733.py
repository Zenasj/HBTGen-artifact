import torch
import torch.nn as nn
import torch._dynamo

model = nn.Linear(5, 5)

def new_forward(*args, **kwargs):
    return nn.Linear.forward(model, *args, **kwargs)

model.forward = new_forward

fn = torch.compile(model, backend='eager')
fn(torch.randn(1, 5))

model = nn.Linear(5, 5)

def new_forward1(*args, **kwargs):
    return nn.Linear.forward(model, *args, **kwargs)

print("Is method originally:", inspect.ismethod(model.forward))  # True
model.forward = new_forward1

old_forward = model.forward
print("Is method now:", inspect.ismethod(old_forward))   # False

def new_forward2(*args, **kwargs):
    return old_forward(*args, **kwargs)
    
model.forward = new_forward2
 
fn = torch.compile(model, backend='eager')
fn(torch.randn(1, 5))

model = nn.Linear(5, 5)
print("Is method originally:", inspect.ismethod(model.forward))   # True

model = torch.compile(model, backend='eager')
old_forward = model.forward
# model.forward is no longer a method after doing torch.compile
print("Is method now:", inspect.ismethod(old_forward))  # False

def new_forward(*args, **kwargs):
    return old_forward(*args, **kwargs)

model.forward = new_forward

fn = torch.compile(model, backend='eager')
fn(torch.randn(1, 5))