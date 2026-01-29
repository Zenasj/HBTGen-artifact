# torch.rand(100, 16, 744, 744, dtype=torch.float32)  # B=100, C=16, H=744, W=744
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=31,
            stride=1,
            padding=15,
            padding_mode="circular",
            bias=False
        )
        # Initialize weights as in the original code
        nn.init.normal_(self.conv.weight)  # Matches torch.randn initialization

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(100, 16, 744, 744, dtype=torch.float32)

