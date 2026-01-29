# torch.rand(B, S, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # [batch_size, sequence_length] → [batch_size, sequence_length]
        a = torch.nn.functional.relu(a)
        return a

def my_model_function():
    return MyModel()

def GetInput():
    # Example input with batch=2, sequence_length=5
    return torch.rand(2, 5, dtype=torch.float32)

