# Input: a tuple of 3 tensors of shape () each → e.g., (torch.rand(()), torch.rand(()), torch.rand(()))
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)  # 3 features from column_stack of 3 scalars

    def forward(self, tensors):
        stacked = torch.column_stack(tensors)
        stacked = stacked.view(1, -1)  # Reshape to (1, 3)
        return self.linear(stacked)

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(()), torch.rand(()), torch.rand(()))

