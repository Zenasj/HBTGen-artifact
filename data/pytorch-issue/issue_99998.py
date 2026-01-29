# torch.rand(B, 36, 1, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 200, 2, batch_first=True)
    
    def forward(self, x):
        return self.lstm(x)[0]  # Return outputs only for simplicity

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(20, 36, 1, dtype=torch.float)  # Matches batch_first LSTM input requirements

