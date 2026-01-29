# torch.rand(1, 24, 6, 6, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=24,
            out_channels=24,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=24,
            bias=False,
            padding_mode='zeros'  # Explicitly set to match original issue
        )

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    model = MyModel()
    with torch.no_grad():
        model.conv.weight.fill_(1.0)  # Replicate original weight initialization
    return model

def GetInput():
    return torch.rand(1, 24, 6, 6, dtype=torch.float32)

