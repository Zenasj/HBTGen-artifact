# torch.rand(10, 1, 100, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(100, 256, bias=False)  # Matches decoder's input dim (256)
        self.rnn = nn.LSTM(256, 256, 1)  # LSTM layer with input/hidden size 256
        self.decoder = nn.Linear(256, 33278)  # Output dim from user's state_dict example

    def forward(self, x):
        # x: (seq_len, batch, input_dim=100)
        x = self.encoder(x)  # -> (seq_len, batch, 256)
        x, _ = self.rnn(x)   # -> (seq_len, batch, 256)
        x = self.decoder(x)  # -> (seq_len, batch, 33278)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Returns (seq_len=10, batch=1, input_dim=100)
    return torch.rand(10, 1, 100, dtype=torch.float)

