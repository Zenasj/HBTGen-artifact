# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import einops

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv(x)
        # Use einops rearrange to modify tensor shape
        x = einops.rearrange(x, 'b c h w -> b h w c')
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 32, 32, dtype=torch.float32)

