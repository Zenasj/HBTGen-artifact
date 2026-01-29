# torch.rand(B, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.input = nn.Linear(2, 2)
        self.hiddenMinusLast = nn.ModuleList([nn.Linear(2, 2)])  # One hidden layer in ModuleList
        self.hiddenLast = nn.Linear(2, 2)
        self.output = nn.Linear(2, 2)

    def forward(self, x):
        out = self.activation(self.input(x))
        for layer in self.hiddenMinusLast:
            out = self.activation(layer(out))
        out = self.activation(self.hiddenLast(out))
        out = self.output(out)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random tensor matching the model's input requirements
    return torch.randn(4, 2, dtype=torch.float32)  # Batch size 4, 2 features

