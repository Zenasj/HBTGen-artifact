# torch.rand(5, 5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Encapsulate all discussed transforms as submodules and return their outputs
        vmap_out = torch.func.vmap(torch.sum)(x)               # vmap transform
        grad_out = torch.func.grad(torch.sum)(x)               # grad transform
        jacrev_out = torch.func.jacrev(torch.sum)(x)           # jacrev transform
        jacfwd_out = torch.func.jacfwd(torch.sum)(x)           # jacfwd transform
        return vmap_out, grad_out, jacrev_out, jacfwd_out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 5)  # Matches input shape from original issue example

