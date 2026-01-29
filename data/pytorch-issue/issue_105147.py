# torch.rand(B, 1, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        # Reshape 4D input to 2D (batch, features)
        x = x.view(x.size(0), -1)
        return self.linear(x)

def my_model_function():
    # Initialize model with CUDA device (matches original issue's setup)
    return MyModel().cuda()

def GetInput():
    # Generate 4D tensor matching the model's expected input shape
    return torch.rand(1, 1, 1, 1, dtype=torch.float32).cuda()

