# torch.rand(seq_len, batch, input_size, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(10, 10)  # Matches the LSTM(10, 10) from the issue's reproduction code
    
    def forward(self, x):
        return self.lstm(x)[0]  # Return outputs (ignoring hidden state for simplicity)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 2, 10, dtype=torch.float32)  # seq_len=5, batch=2, input_size=10

