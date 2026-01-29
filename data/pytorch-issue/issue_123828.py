# torch.rand(1, 1, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # LSTM with hidden_size=128 (problematic case causing deadlock)
        self.lstm_large = nn.LSTM(input_size=2, hidden_size=128, num_layers=1, batch_first=True)
        # LSTM with hidden_size=6 (non-problematic case for comparison)
        self.lstm_small = nn.LSTM(input_size=2, hidden_size=6, num_layers=1, batch_first=True)
    
    def forward(self, x):
        # Run both models and return their outputs for comparison
        out_large, _ = self.lstm_large(x)
        out_small, _ = self.lstm_small(x)
        return out_large, out_small

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 2, dtype=torch.float32)

