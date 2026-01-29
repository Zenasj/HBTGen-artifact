# torch.rand(B, 1, 16, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=1,
            batch_first=True,
            dropout=0.1
        )

    def forward(self, x):
        return self.encoder_layer(x)

def my_model_function():
    model = MyModel()
    model.eval()  # Matches original issue's eval() setup
    return model

def GetInput():
    return torch.randn(4, 1, 16, device=torch.device("cuda"), dtype=torch.float32)

