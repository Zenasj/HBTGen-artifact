# torch.rand(B, 3, 1, 6, dtype=torch.float32)

import torch
from torch import nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        output = self.linear(recurrent)
        return output

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(3, 4, 4),
            BidirectionalLSTM(4, 4, 4)
        )
    
    def forward(self, x):
        x2 = torch.squeeze(x, 2)        # Squeeze spatial dimension (size 1)
        x3 = torch.permute(x2, (0, 2, 1))  # Permute to (batch, time_steps, features)
        out = self.SequenceModeling(x3)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 1, 6)  # Matches input requirements after processing

