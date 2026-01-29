# torch.rand(B, C, L, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_channels=100, output_channels=50):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(output_channels)  # LayerNorm over transposed channel dimension

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()  # Ensure contiguous after transpose
        x = self.norm(x)
        x = x.transpose(1, 2).contiguous()  # Restore original channel dimension order
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 100, 25, dtype=torch.float32)  # Matches input shape (B, C, L)

