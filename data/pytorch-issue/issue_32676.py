# torch.rand(1, 12, 80, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lstm = nn.LSTM(
            80,          # input_size
            48,          # hidden_size
            4,           # num_layers
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
    
    def forward(self, x):
        y, _ = self.lstm(x)
        return y

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 12, 80, dtype=torch.float32)

