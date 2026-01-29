# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Reproduces the norm operation causing gradient None issue with torch.compile
        return torch.masked.norm(x, 0, mask=None)

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Returns a 4D tensor with requires_grad=True to trigger backward
    return torch.rand(1, 1, 1, 1, dtype=torch.float32, requires_grad=True)

