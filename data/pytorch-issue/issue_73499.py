# torch.rand(B, 5, 1, 1, dtype=torch.float16)  # Shape inferred from 2x5 example; dtype can be float16 or bfloat16
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute mean via std_mean and direct mean, return their absolute difference
        mean_std, _ = torch.std_mean(x)
        mean_direct = x.mean()
        return torch.abs(mean_std - mean_direct)

def my_model_function():
    return MyModel()

def GetInput():
    B, C, H, W = 2, 5, 1, 1  # Matches 2x5 example shape as 4D tensor
    dtypes = [torch.float16, torch.bfloat16]
    import random
    dtype = random.choice(dtypes)  # Randomly choose between the two dtypes
    return torch.rand(B, C, H, W, dtype=dtype)

