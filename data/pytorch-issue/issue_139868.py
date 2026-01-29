import torch
from torch import nn

# torch.rand(B, C, H, W, dtype=torch.float32)  # Input shape (1, 3, 64, 64)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Conv2d(3, 64, 3, bias=True)
    
    def forward(self, x: torch.Tensor):
        x = self.fc0(x)
        return x[:, 2:, :-1]  # Failing slice as per the issue

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((1, 3, 64, 64), dtype=torch.float32)

