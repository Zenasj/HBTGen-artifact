from torch import nn
import torch
from functools import reduce
from operator import mul

# torch.rand(B, C, H, W, dtype=torch.float32)  # Input shape: (B, 3, 64, 64)
class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

def flat_softmax(inp):
    orig_size = inp.size()
    flat = inp.view(-1, reduce(mul, orig_size[2:]))
    flat = torch.nn.functional.softmax(flat, -1)
    return flat.view(*orig_size)

def dsnt(heatmaps):
    batch_size, n_points, height, width = heatmaps.size()
    x = torch.linspace(0, width-1, width).to(heatmaps.device)
    y = torch.linspace(0, height-1, height).to(heatmaps.device)
    x_coords = (heatmaps * x.view(1, 1, 1, -1)).sum(-1)
    y_coords = (heatmaps * y.view(1, 1, -1, 1)).sum(-2)
    coords = torch.stack([x_coords, y_coords], dim=-1)
    return coords

class MyModel(nn.Module):
    def __init__(self, n_locations):
        super().__init__()
        self.fcn = FCN()
        self.hm_conv = nn.Conv2d(16, n_locations, kernel_size=1, bias=False)

    def forward(self, images):
        fcn_out = self.fcn(images)
        unnormalized_heatmaps = self.hm_conv(fcn_out)
        heatmaps = flat_softmax(unnormalized_heatmaps)
        coords = dsnt(heatmaps)
        return coords, heatmaps

def my_model_function():
    return MyModel(n_locations=10)

def GetInput():
    return torch.rand(1, 3, 64, 64, dtype=torch.float32)

