import torch.nn as nn

import torch
import torchdynamo
import torchinductor
from torchinductor import config

torchinductor.config.debug = True

def relu(x0):
    return torch.nn.functional.relu(x0)

def maximum(x0, x1):
    return (torch.clamp_min(x0, 0.5), torch.maximum(x0, x1))

def clamp_min_tensor(x0, x1):
    return torch.clamp_min(x0, x1)

device='cuda'
dtype=torch.half
x0 = torch.arange(-3, 4, 1, device=device, dtype=dtype)
x1 = torch.zeros_like(x0)
x0[1]=float('nan')
x0[-1]=float('nan')
x1[2]=float('nan')
print(x0)
optimize_ctx = torchdynamo.optimize("inductor")
with optimize_ctx:
    out_inductor = (relu(x0), maximum(x0, x1))
out_eager = (relu(x0), maximum(x0, x1)) 
print(out_inductor) #clamp_min doesn't propagate nans, maximum propagates nans only from one of the args
print(out_eager)
x0=torch.randint(4,(7,), device=device)
with optimize_ctx:
    out_inductor=maximum(x0, x1)
out_eager=maximum(x0, x1)
print(out_inductor) #clamp_min doesn't type promote
print(out_eager)
x0 = torch.randn(7, device=device, dtype=dtype)
with optimize_ctx:
    out_inductor=clamp_min_tensor(x0, x1) #errors out