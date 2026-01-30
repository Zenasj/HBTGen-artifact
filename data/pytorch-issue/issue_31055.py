import torch.nn as nn
import torch.nn.functional as F

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.interpolate(x, size=(8,8), mode="bilinear", align_corners=False)