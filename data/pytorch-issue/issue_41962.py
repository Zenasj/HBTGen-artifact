# torch.rand(B, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1:
            return torch.tensor(1.0, dtype=x.dtype, device=x.device)
        else:
            return torch.tensor(3.0, dtype=x.dtype, device=x.device)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 2)

