# torch.rand(2, 2, 1, dtype=torch.long)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.index = torch.tensor([[[1], [1]], [[1], [1]]], dtype=torch.long)

    def forward(self, x):
        return x[self.index]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 2, (2, 2, 1), dtype=torch.long)

