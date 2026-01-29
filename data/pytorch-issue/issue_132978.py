# torch.rand(B, 8, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.linears = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Linear(dim, dim, bias=False),
            nn.Linear(dim, dim, bias=False)
        )
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linears(x)
        y = self.proj(y)
        return y

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 8, dtype=torch.float)

