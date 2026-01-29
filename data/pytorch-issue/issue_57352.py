# torch.rand(B, D_in, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

def my_model_function():
    # Initialize with parameters from the original repro example (D_in=1000, H=100, D_out=10)
    return MyModel(1000, 100, 10)

def GetInput():
    # Matches input shape (N=64, D_in=1000) from the issue's example
    return torch.rand(64, 1000, dtype=torch.float32)

