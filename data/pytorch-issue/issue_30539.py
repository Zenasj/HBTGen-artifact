# torch.rand(B, S, 10, dtype=torch.float)  # Input shape: (batch, sequence_length, input_size=10)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.rnn = nn.GRU(input_size=10, hidden_size=10, batch_first=True)
    
    def forward(self, x):
        x, _ = self.rnn(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Example batch size
    S = 5  # Example sequence length
    return torch.rand(B, S, 10, dtype=torch.float)

