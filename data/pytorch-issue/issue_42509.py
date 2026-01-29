# torch.rand(B=1, C=1, H=5, W=1, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Use valid output_size (1,1) to avoid segmentation fault
        return F.adaptive_avg_pool2d(x, (1, 1))

def my_model_function():
    return MyModel()

def GetInput():
    # Matches expected 4D input (B, C, H, W)
    return torch.rand(1, 1, 5, 1, dtype=torch.float32)

