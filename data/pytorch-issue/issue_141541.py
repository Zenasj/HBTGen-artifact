# torch.rand(B, C, D, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=32)
    
    def forward(self, x):
        return self.gn(x)

def my_model_function():
    model = MyModel()
    model.eval()  # Matches original issue's model setup
    return model

def GetInput():
    return torch.randn(1, 32, 128, 128, 128)  # Matches input shape from original reproduction code

