# torch.rand(B, C, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1)

    def forward(self, x):
        return self.maxpool(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # The input is a tensor of shape (B, C, H, W) with negative values
    B, C, H, W = 2, 2, 64, 64
    return -torch.rand(B, C, H, W, dtype=torch.float32)

