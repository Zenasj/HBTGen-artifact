# torch.rand(2, 4, 5, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # LSTM with input_size=5, hidden_size=5, bidirectional=True, batch_first=True
        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=5,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
    
    def forward(self, x):
        # Returns only the output tensor (ignores hidden states)
        return self.lstm(x)[0]

def my_model_function():
    # Initialize LSTM with fixed seed for reproducibility
    torch.manual_seed(1234)
    return MyModel()

def GetInput():
    # Random input matching batch_first=True (batch, seq_len, features)
    return torch.randn(2, 4, 5, dtype=torch.float)

