# torch.randint(0, 1000, (1,), dtype=torch.long)  # Input is a single integer token
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, token: torch.Tensor) -> torch.Tensor:
        token_str = str(token.item())
        return torch.tensor([ord(c) for c in token_str], dtype=torch.int)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 1000, (1,), dtype=torch.long)

