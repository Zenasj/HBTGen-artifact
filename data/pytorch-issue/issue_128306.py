# torch.rand(1, 4, dtype=torch.bfloat16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, score_thresh):
        super().__init__()
        self.register_buffer('score_thresh', torch.tensor(score_thresh, dtype=torch.bfloat16))
    
    def forward(self, x):
        keep = x > self.score_thresh
        return torch.nonzero(keep)

def my_model_function():
    # Initialize with the threshold from the example (0.2)
    return MyModel(score_thresh=0.2)

def GetInput():
    # Generate random input matching the example's shape and dtype
    return torch.rand(1, 4, dtype=torch.bfloat16)

