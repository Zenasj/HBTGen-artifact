# torch.rand(B, S, D, dtype=torch.float32)  # B=32, S=20, D=64
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            bias=False,
            batch_first=True
        ).eval()  # Matches original issue's configuration

    def forward(self, x):
        return self.encoder_layer(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(32, 20, 64)  # Matches original input dimensions (batch_size, seqlen, d_model)

