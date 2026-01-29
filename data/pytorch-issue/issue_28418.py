# torch.rand(seq_len, batch_size, 32, dtype=torch.float32)  # Input shape (seq_len, batch, input_size)
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

class MyModel(nn.Module):
    def __init__(self, input_size=32, hidden_size=64, num_layers=1, batch_first=False):
        super(MyModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=batch_first)

    def forward(self, input, hx=None):
        # Handle PackedSequence explicitly to avoid problematic code paths in RNN's forward
        if isinstance(input, PackedSequence):
            input, _ = pad_packed_sequence(input, batch_first=self.rnn.batch_first)
        return self.rnn(input, hx)

def my_model_function():
    return MyModel(input_size=32, hidden_size=64, num_layers=1, batch_first=False)

def GetInput():
    # Default shape matching RNN(32,64,1) with batch_first=False (seq_len, batch, input_size)
    return torch.rand(5, 2, 32, dtype=torch.float32)  # Example input dimensions

