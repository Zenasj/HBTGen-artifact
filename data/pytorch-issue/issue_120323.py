# torch.rand(B, 3, 64, 64, dtype=torch.float32)  # Input shape inferred from Stable Diffusion XL's typical image input
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Replicates the problematic Conv2d layer from AutoencoderKL's encoder initialization
        self.conv_in = nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1)  # Common parameters for SD-XL autoencoder

    def forward(self, x):
        return self.conv_in(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Batch size 2, 3 channels, 64x64 image (common SD-XL resolution)
    return torch.rand(2, 3, 64, 64, dtype=torch.float32)

