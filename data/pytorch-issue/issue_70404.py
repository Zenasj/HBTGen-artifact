# torch.rand(B, T, D, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(10, 10, 1, batch_first=True)  # batch_first to match input shape
        self.linear = nn.Linear(10, 1)  # Fused as per comparison in the issue
        
    def forward(self, x):
        # Process GRU and Linear in parallel for comparison
        gru_out, _ = self.gru(x)
        linear_input = x[:, -1, :]  # Extract last time step for Linear compatibility
        linear_out = self.linear(linear_input)
        return (gru_out, linear_out)  # Return both outputs for comparison

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape (batch, seq_len, features) for GRU and Linear compatibility
    return torch.randn(4, 20, 10, dtype=torch.float)

