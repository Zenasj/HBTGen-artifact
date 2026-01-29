# torch.rand(B, C, F, H, W, dtype=torch.half, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.spatial_conv = nn.Conv2d(*args, **kwargs)
        new_args = (args[1], *args[1:])  # Adjust args to match Conv1d's expected signature
        d1_kwargs = dict(kwargs)
        d1_kwargs['stride'] = 1  # Enforce stride=1 for temporal_conv as in original code
        self.temporal_conv = nn.Conv1d(*new_args, **d1_kwargs)
        # Initialize central kernel slice of temporal_conv to identity matrix
        kernel_size = new_args[-1]
        mid_idx = (kernel_size - 1) // 2
        torch.nn.init.eye_(self.temporal_conv.weight[:, :, mid_idx])

    def forward(self, input):
        B, C, F, H, W = input.size()
        x = input.permute(0, 2, 3, 4, 1)  # B,F,H,W,C
        x = x.contiguous().view(-1, H, W, C).permute(0, 3, 1, 2)  # (B*F, C, H, W)
        x = self.spatial_conv(x)  # Apply spatial convolution
        x = x.view(B, F, -1, H, W)  # B,F,out_C,H,W
        x = x.permute(0, 3, 4, 1, 2)  # B,H,W,F,out_C
        out_channel = x.size(4)
        x = x.contiguous().view(-1, F, out_channel).permute(0, 2, 1)  # (B*H*W, out_C, F)
        x = self.temporal_conv(x)  # Apply temporal convolution
        x = x.view(B, H, W, -1, F)  # B,H,W,out_C',F
        x = x.permute(0, 3, 4, 1, 2)  # Final output: B,out_C',F,H,W
        return x

def my_model_function():
    # Initialize with parameters from the original issue's example
    return MyModel(320, 320, 3, padding=1).cuda().half()

def GetInput():
    # Generate input matching the model's expected dimensions and dtype
    return torch.rand((4, 320, 16, 256, 256), dtype=torch.half, device='cuda')

