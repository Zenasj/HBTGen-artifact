# torch.rand(B, C, L, dtype=torch.float32)  # B: batch_size, C: in_channels, L: length
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        in_channels = 64
        out_channels = 128
        scale_factor = 8
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=scale_factor * 2, stride=scale_factor)
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x).contiguous()
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 8
    in_channels = 64
    length = 16
    return torch.randn(batch_size, in_channels, length, dtype=torch.float32).contiguous()

