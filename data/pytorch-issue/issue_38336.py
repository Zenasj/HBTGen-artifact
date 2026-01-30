import torch

bash
@torch.jit.script
def mm(a,b):
    a %= b
    return a

bash
@torch.jit.script 
def mm(a,b): 
    a.fmod_(b) 
    return a