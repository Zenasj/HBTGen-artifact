# torch.rand(B, 100, 8, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.rnn1 = nn.LSTM(8, 8, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(8 * 2, 8)
        self.rnn2 = nn.LSTM(8, 8, bidirectional=True, batch_first=True)
        self.linear2 = nn.Linear(8 * 2, 8)

    def forward(self, input):
        rnn_output1, _ = self.rnn1(input)
        linear_output1 = self.linear1(rnn_output1)
        rnn_output2, _ = self.rnn2(linear_output1)
        linear_output2 = self.linear2(rnn_output2)
        return linear_output2

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 100, 8, dtype=torch.float32)

