# torch.rand(B, 2, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
    
    def forward(self, x):
        # Flatten 4D input (B, C, H, W) to 2D (B, C*H*W) for linear layer
        return self.linear(x.view(x.size(0), -1))

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the model's expected input shape (B=10, C=2, H=1, W=1)
    return torch.rand(10, 2, 1, 1, dtype=torch.float32)

