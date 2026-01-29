# torch.rand(B, 3, 128, 128, dtype=torch.float32)
import torch
import torch.nn as nn

class WindowAttention(nn.Module):
    def forward(self, x):
        return x  # Dummy implementation to mimic the original

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 128, kernel_size=4, stride=4)  # Matches patch_size=4
        self.window_attn = WindowAttention()
        # Compile the problematic submodule's forward method as in the original issue
        self.window_attn.forward = torch.compile(self.window_attn.forward)
        # Add minimal dummy layers to mimic SwinTransformerV2 structure
        self.layers = nn.Sequential(
            nn.Linear(128, 128),  # Stub for transformer blocks
            nn.ReLU()
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.window_attn(x)
        # Dummy forward pass continuation
        x = self.layers(x.mean(dim=(-2, -1)))  # Global average pool + linear
        return x

def my_model_function():
    # Disable DDP optimization in Dynamo to avoid the error (as per user fix)
    torch._dynamo.config.optimize_ddp = False
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 128, 128, dtype=torch.float32)

