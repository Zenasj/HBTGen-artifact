# torch.rand(B, 10, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        output = self.linear(recurrent)
        return output

def my_model_function():
    return MyModel(10, 10, 10)

def GetInput():
    B = 2  # Arbitrary batch size (matches issue example's 16, but 2 is minimal)
    return torch.rand(B, 10, 10, dtype=torch.float32)

