# torch.rand(B, C, H, W, dtype=torch.float32)  # Input shape: (1, 1, 4, 2)
import torch
import torch.nn as nn
from torch.nn import functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_h = 2.05  # Original scale factor for height
        self.scale_w = 3.15  # Original scale factor for width

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]
        out_h = int(round(input_h * self.scale_h))
        out_w = int(round(input_w * self.scale_w))

        # Create output grid coordinates
        oy = torch.arange(out_h, device=x.device)
        ox = torch.arange(out_w, device=x.device)
        oy_grid, ox_grid = torch.meshgrid(oy, ox, indexing='ij')

        # Method 1: Floor-based indexing (matches first proposed method)
        i_y1 = (oy_grid.float() / self.scale_h).floor().int()
        i_x1 = (ox_grid.float() / self.scale_w).floor().int()
        i_y1 = torch.clamp(i_y1, 0, input_h - 1)
        i_x1 = torch.clamp(i_x1, 0, input_w - 1)
        out1 = x[0, 0, i_y1, i_x1].unsqueeze(0).unsqueeze(0)

        # Method 2: Round-based indexing (matches second proposed method)
        f_y = (oy_grid.float() + 0.5) / self.scale_h - 0.5
        f_x = (ox_grid.float() + 0.5) / self.scale_w - 0.5
        i_y2 = torch.round(f_y).int()
        i_x2 = torch.round(f_x).int()
        i_y2 = torch.clamp(i_y2, 0, input_h - 1)
        i_x2 = torch.clamp(i_x2, 0, input_w - 1)
        out2 = x[0, 0, i_y2, i_x2].unsqueeze(0).unsqueeze(0)

        return out1, out2

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 1, 4, 2)

