import torch
import torch.nn as nn

# torch.rand(B, 120, 10, 20, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=120,
            out_channels=100,
            kernel_size=(3, 8),
            groups=2
        )  # Uses default kaiming_uniform_ init with a=math.sqrt(5)

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Returns a Conv2d model with the configuration from the issue's example
    return MyModel()

def GetInput():
    # Returns random input tensor matching (B, C, H, W) = (2, 120, 10, 20)
    return torch.rand(2, 120, 10, 20, dtype=torch.float32)

