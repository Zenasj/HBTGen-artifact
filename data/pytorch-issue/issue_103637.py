# torch.randint(0, 256, (4,), dtype=torch.uint8)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x >> 2).to(torch.long)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 256, (4,), dtype=torch.uint8)

