# torch.rand(1, 1, 128, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.gru = nn.GRU(128, 128, 1, bias=True)  # Matches the model configuration in the issue's example

    def forward(self, x):
        return self.gru(x)  # Forward pass with default hidden state (None)

def my_model_function():
    return MyModel()  # Returns the GRU model instance

def GetInput():
    return torch.rand(1, 1, 128, dtype=torch.float32)  # Matches input shape (seq_len=1, batch=1, input_size=128)

