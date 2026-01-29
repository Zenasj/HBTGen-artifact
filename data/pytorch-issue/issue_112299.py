# torch.rand(1, 3, 1, 2, 2, dtype=torch.bfloat16, device='cuda', requires_grad=True)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, x):
        # Replicate the original comparison setup
        b = x.detach()[:, :, 0].clone().requires_grad_(True)
        trilinear_out = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")
        bilinear_out = F.interpolate(b, scale_factor=2, mode="bilinear")
        return trilinear_out, bilinear_out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 1, 2, 2, dtype=torch.bfloat16, device='cuda', requires_grad=True)

