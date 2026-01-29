import torch
from torch import nn
from einops import rearrange

# torch.rand(B, C, H, W, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.conv(x)
        # Einops rearrange triggers backend lookup causing the graph break
        x = rearrange(x, 'b c h w -> b h w c')
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 32, 16, 16, dtype=torch.float32)

