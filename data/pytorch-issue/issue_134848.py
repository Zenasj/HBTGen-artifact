# torch.rand(B, C, H, W, dtype=torch.float16)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute mean along dim=0 (batch) with keepdim=True
        # Output shape: (1, C, H, W)
        out_shape = (1,) + x.shape[1:]
        y = torch.empty(out_shape, dtype=torch.float16, device=x.device)
        torch.mean(x, dim=0, keepdim=True, out=y)
        return y

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape (B=3, C=1, H=1, W=4) as per issue's example
    return torch.rand(3, 1, 1, 4, dtype=torch.float16)

