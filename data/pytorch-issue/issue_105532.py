# torch.rand(B, S, C, dtype=torch.float32, device='cuda')  # B=batch, S=seq_len, C=input_size
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
    
    def forward(self, x):
        self.lstm.flatten_parameters()  # Line causing Dynamo Unsupported error
        out, _ = self.lstm(x)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    B, S, C = 2, 10, 256  # Batch, sequence length, input features
    return torch.rand(B, S, C, dtype=torch.float32, device='cuda')

