# torch.rand(20, 10, dtype=torch.float32).cuda()  # Input shape (seq_len, input_size) with batch_size=1
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.lstm = nn.LSTM(10, 1)  # input_size=10, hidden_size=1

    def forward(self, x):
        x = self.linear(x)
        x, _ = self.lstm(x)
        return x

def my_model_function():
    model = MyModel()
    model.cuda()  # Explicitly move model to CUDA device
    return model

def GetInput():
    # Returns a 2D tensor (seq_len, input_size) with batch_size=1 implicit
    return torch.randn(20, 10, dtype=torch.float32).cuda()

