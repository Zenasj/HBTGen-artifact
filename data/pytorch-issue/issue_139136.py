# torch.rand(2, 3, 10, dtype=torch.float32), torch.randint(5, (2, 3), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, C=10, V=5):
        super().__init__()
        self.C = C
        self.V = V
        self.linear = nn.Linear(C, V)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs):
        x, y = inputs
        return self.ce(self.linear(x).view(-1, self.V), y.view(-1))

def my_model_function():
    return MyModel()

def GetInput():
    B, T = 2, 3  # Example batch and sequence lengths
    C, V = 10, 5  # Input features and output classes
    x = torch.rand(B, T, C, dtype=torch.float32)
    y = torch.randint(V, (B, T), dtype=torch.long)
    return (x, y)

