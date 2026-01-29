# torch.rand(B, 2, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1, bias=False),
        )
    
    def forward(self, x):
        # Flatten 4D input to 2D (B, C*H*W) for Linear layers
        x = x.view(x.size(0), -1)
        return self.layers(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 1, 1, dtype=torch.float32)

