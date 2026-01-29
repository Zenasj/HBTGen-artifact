# torch.rand(1, 960, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(960, 480, bias=True)  # Input size 960 → hidden size 480

    def forward(self, x):
        # LSTM expects (seq_len, batch, input_size), but input here is 2D (seq_len=1, batch=1)
        output, (hn, cn) = self.lstm(x.unsqueeze(0))  # Add seq_len dimension
        return output.squeeze(0)  # Remove seq_len for compatibility with original code's shape

def my_model_function():
    return MyModel()

def GetInput():
    # Original input shape (1, 960) but LSTM requires 3D tensor → auto-adjust in forward()
    return torch.rand(1, 960, dtype=torch.float32)

