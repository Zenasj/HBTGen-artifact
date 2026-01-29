# torch.rand(B, C, H, W, dtype=torch.float32)  # B=batch=1, C=input_size=6, H=seq_len=1, W=1 (unused)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=6, hidden_size=3)
        
    def forward(self, input, h_0=None):
        # The RNN expects 'hx' parameter, but we use 'h_0' to match documentation intent
        return self.rnn(input, hx=h_0)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 3D tensor (seq_len=1, batch=1, input_size=6) matching RNN input requirements
    return torch.rand(1, 1, 6, dtype=torch.float32)

