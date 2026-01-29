# torch.rand(B, C, L, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=16, out_features=16)
        self.avg = nn.AdaptiveAvgPool1d(output_size=[0])  # Output size 0 as in original code

    def forward(self, x):
        x = self.fc(x)
        x = self.avg(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(16, 16, 16, dtype=torch.float32)

