# torch.rand(B, C, H, W, dtype=torch.float32)  # B=500, C=1, H=32, W=74
import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Interpolate input with align_corners=False
        input2 = F.interpolate(x, scale_factor=5, mode='bilinear', align_corners=False)
        N, C, H, W = x.shape
        
        # Generate grid as per example parameters
        grid = torch.rand(N, 18, 17, 2, device=x.device, dtype=x.dtype) * 1.8 - 0.9

        # Original grid_sample comparison
        output = F.grid_sample(x, grid)
        output2 = F.grid_sample(input2, grid)
        diff_v1 = torch.abs(output2 - output).mean()

        # Adjusted grid coordinates (from grid_sample_v2)
        grid_x = grid[..., 0]
        grid_y = grid[..., 1]
        grid_x_adj = ((grid_x + 1)/2 * W - 0.5)/(W-1)*2 - 1
        grid_y_adj = ((grid_y + 1)/2 * H - 0.5)/(H-1)*2 - 1
        newgrid = torch.stack([grid_x_adj, grid_y_adj], dim=-1)
        
        # Adjusted grid_sample comparison
        output_v2 = F.grid_sample(x, newgrid)
        output2_v2 = F.grid_sample(input2, newgrid)
        diff_v2 = torch.abs(output2_v2 - output_v2).mean()

        # Return boolean indicating if adjusted method meets threshold
        return (diff_v2 < 1e-6).float()  # 1.0 = success (small difference)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(500, 1, 32, 74, dtype=torch.float32)

