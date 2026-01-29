# torch.rand(B, C, H, W, dtype=torch.float64)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Use output_size=(2,3) based on input spatial dimensions (5x7) and kernel_size 2x2
        # Avoid passing _random_samples to let PyTorch generate it internally
        return F.fractional_max_pool2d(x, kernel_size=(2,2), output_size=(2,3))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 5, 7, dtype=torch.float64)

