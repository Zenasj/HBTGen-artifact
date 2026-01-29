# Input shapes: X (1, 1, 128), h (1, 1, 128)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 128)
        self.gru = nn.GRU(input_size=128, hidden_size=128)

    def forward(self, inputs):
        X, h = inputs
        X = self.linear(X)
        X, h_n = self.gru(X, h)
        return X

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(1, 1, 128), torch.rand(1, 1, 128))

