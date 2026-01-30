import torch

def fn1(x,y,z):
    a = x+y+z     
    return torch.sigmoid(a)