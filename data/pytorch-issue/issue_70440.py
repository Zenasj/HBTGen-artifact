# torch.rand(B, C, D, H, W, dtype=torch.float32)  # Input shape inferred from example (2, 768, 1, 3, 3)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.ConvTranspose3d(
            in_channels=768, out_channels=768,
            kernel_size=[2, 1, 1], dilation=[7, 1, 1], output_padding=0
        )
        self.layer_norm = nn.LayerNorm(768, eps=0.1)
    
    def forward(self, x):
        x = self.conv(x)
        # Permute and contiguous() as in original example to match error scenario
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        return self.layer_norm(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape (B, C, D, H, W) from the original example (2, 768, 1, 3, 3)
    return torch.rand(2, 768, 1, 3, 3, dtype=torch.float32)

