# torch.rand(B, seq_length, input_size, dtype=torch.double)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        # Initialize hidden states on same device/dtype as input
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, 
                        device=x.device, dtype=x.dtype)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, 
                        device=x.device, dtype=x.dtype)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

def my_model_function():
    # Create model with default parameters (matches original example)
    return MyModel(input_size=10, hidden_size=16, num_layers=2).double()

def GetInput():
    # Random input tensor matching (batch, sequence, features) format
    return torch.rand(2, 5, 10, dtype=torch.double)

