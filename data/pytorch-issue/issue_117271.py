# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Extract scalar via item() to trigger dynamic scalar codegen path
        val = x[0, 0, 0, 0].item()
        if val > 0:
            return x * 2
        else:
            return x * 3

def my_model_function():
    return MyModel()

def GetInput():
    # Returns 4D tensor (B=1, C=1, H=1, W=1) to match model's expected input
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

