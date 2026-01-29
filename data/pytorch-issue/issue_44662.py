# torch.rand(B, C, H, W, dtype=torch.float32, requires_grad=True)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        output_size = (1, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size)
    
    def forward(self, x):
        # Returns outputs of both adaptive pooling functions for comparison
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)[0]  # [0] to ignore indices in max_pool output
        return avg_out, max_out  # Return tuple for joint gradcheck analysis

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 3, 4, dtype=torch.float32, requires_grad=True)

