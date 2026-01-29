# torch.rand(1, 1, 62, 62, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution layer with kernel size matching input spatial dimensions (62x62) to compute sum
        self.conv = nn.Conv2d(1, 1, kernel_size=(62, 62), stride=1, padding=0, bias=False)
        self.conv.weight.data.fill_(1.0)  # Initialize kernel to all ones
        self.conv.weight.requires_grad_(False)  # Fixed kernel
        
    def forward(self, x):
        x_squared = x.pow(2)
        conv_out = self.conv(x_squared)
        sum_out = x_squared.sum()
        # Compare convolution result (sum via kernel) with explicit sum
        diff = torch.abs(conv_out - sum_out)
        # Return True if within 0.0001 tolerance (float32 precision limit)
        return torch.all(diff < 1e-4)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 62, 62, dtype=torch.float32)

