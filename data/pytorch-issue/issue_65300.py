# torch.rand(seq_len, batch_size, input_size, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=3)  # Matches issue's model configuration

    def forward(self, x):
        return self.lstm(x)[0]  # Return outputs (ignoring hidden state for simplicity)

def my_model_function():
    # Returns LSTM model instance with default initialization
    return MyModel()

def GetInput():
    # Generates input tensor matching LSTM's expected dimensions (seq_len=5, batch=1, input_size=3)
    return torch.rand(5, 1, 3, dtype=torch.float32)

