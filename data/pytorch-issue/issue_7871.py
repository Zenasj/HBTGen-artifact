# torch.rand(2, 3, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.rnn = nn.RNN(input_size=4, hidden_size=2, num_layers=1, bias=False)
        self.lstm = nn.LSTM(input_size=4, hidden_size=2, num_layers=1, bias=False)
        self.gru = nn.GRU(input_size=4, hidden_size=2, num_layers=1, bias=False)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(x)
        return rnn_out, lstm_out, gru_out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 4, dtype=torch.float32)

