# torch.rand(8, 4, 16, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(16, 16)  # Input/hidden size = 16 as in the issue's 'size' variable
    
    def forward(self, x):
        # Forward pass matches GRU usage in the issue (without hidden state for simplicity)
        return self.gru(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(8, 4, 16, dtype=torch.float32)  # seq_len=8, batch_size=4, input_size=16

