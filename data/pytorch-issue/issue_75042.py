# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Use quantizable LSTM with fixed reshape logic
        self.lstm = nn.quantizable.LSTM(input_size=10, hidden_size=20, bidirectional=True)

    def forward(self, x):
        # x is (seq_len, batch, features, 1) → squeeze last dimension to 3D
        x = x.squeeze(-1)  # Shape becomes (H, B, C)
        # Apply LSTM
        out, _ = self.lstm(x)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    # Input dimensions (B=2, C=10, H=5, W=1) → permuted to (H, B, C, W)
    return torch.rand(2, 10, 5, 1, dtype=torch.float32).permute(2, 0, 1, 3)

