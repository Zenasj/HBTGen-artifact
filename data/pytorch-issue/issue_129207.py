# torch.rand(1, 1, 3, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Both layers have out_channels exceeding 65536, triggering the MPS issue
        self.conv1d = nn.Conv1d(1, 65537, kernel_size=3, padding=1)
        self.conv2d = nn.Conv2d(1, 65537, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x):
        # Process Conv1d by squeezing the last dimension (W)
        x_1d = x.squeeze(-1)  # Convert to 3D tensor (N, C, L)
        out1d = self.conv1d(x_1d)
        # Process Conv2d directly (4D tensor)
        out2d = self.conv2d(x)
        return (out1d, out2d)  # Return outputs of both layers for comparison

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 3, 1, dtype=torch.float32)  # Random input matching both layers' requirements

