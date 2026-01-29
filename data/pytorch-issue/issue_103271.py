# torch.rand(B, 128, 14, 14, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(128, 256, 1)  # 1x1 convolution
    
    def forward(self, x):
        x = self.conv(x)
        x_transposed = x.permute(0, 2, 3, 1)  # (B, H, W, 256)
        a = x_transposed[..., :128]
        b = x_transposed[..., 128:]
        c = torch.cat([a, b], dim=-1)  # (B, H, W, 256)
        result = c.permute(0, 3, 1, 2)  # Restore original channel dimension
        return result

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 128, 14, 14, dtype=torch.float32)

