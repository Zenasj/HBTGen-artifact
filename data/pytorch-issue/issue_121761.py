# torch.rand(1, 40, 19, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, size=19, h_dim=40):
        super().__init__()
        self.rnn = nn.GRU(size, h_dim, batch_first=True)
    
    def forward(self, x):
        _, states = self.rnn(x)
        return states

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 40, 19, dtype=torch.float32)

