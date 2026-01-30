py
import torch

@torch.compile
def func(x):
    invalid = (x < 0.0).bool()
    if torch.any(invalid): # Probable offending use of Aten.any
        print("foo")

    return x

x = torch.randn(640,1000,device='cuda')
d = func(x)

py
import torch

@torch.compile
def func(x):
    invalid = (x < 0.0).bool()
    if torch.any(invalid): # Probable offending use of Aten.any
        print("foo")

    return x

x = torch.randn(256,1024,device='cuda')
d = func(x)