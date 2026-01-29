# torch.rand(B, C, H, W, dtype=torch.float32)  # Input shape: (1, 3, 256, 256)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.MultiheadAttention(512, 8, dropout=0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Replicate the original issue's test setup with random tensors
        out = torch.randn([200, 1, 512], device=x.device, dtype=x.dtype)
        out2 = torch.randn([200, 1, 512], device=x.device, dtype=x.dtype)
        out3 = torch.randn([200, 1, 512], device=x.device, dtype=x.dtype)
        out = self.conv3(query=out, key=out, value=out2)[0]
        return out

def my_model_function():
    # Initialize model with the same parameters as the issue's example
    model = MyModel(upscale_factor=3)
    # Manually initialize weights to match state_dict logic (placeholder)
    # Note: Actual weights depend on the missing state_dict, so use random init here
    return model

def GetInput():
    return torch.randn(1, 3, 256, 256, dtype=torch.float32)

