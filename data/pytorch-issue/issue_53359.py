# torch.rand(B, T, C, dtype=torch.float32)  # B: batch size, T: sequence length, C: input size

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MyModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return out

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    input_size = 10  # Example input size
    hidden_size = 20  # Example hidden size
    num_layers = 2  # Example number of layers
    return MyModel(input_size, hidden_size, num_layers)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B = 5  # Batch size
    T = 10  # Sequence length
    C = 10  # Input size
    return torch.rand(B, T, C, dtype=torch.float32)

