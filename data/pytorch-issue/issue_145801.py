# torch.rand(2895, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        mat = torch.outer(x, x)
        return torch.linalg.eigh(mat)

def my_model_function():
    return MyModel()

def GetInput():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.rand(2895, device=device, dtype=torch.float32)

