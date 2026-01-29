# torch.rand(1, 2, 4, 8, 16, dtype=torch.float32).cuda()  # 5D input tensor (B, C, D, H, W)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=(0.25, 0.125, 0.0625),
            mode='trilinear',
            align_corners=False,
            recompute_scale_factor=False
        )

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 4, 8, 16, dtype=torch.float32).cuda()

