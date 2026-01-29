# torch.rand(B, C, H, W, dtype=torch.float32, device='cuda')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Uses torch._refs.mean as per the PR's context for nvFuser compatibility
        return torch._refs.mean(x, keepdim=False)

def my_model_function():
    # Returns an instance of the model using the corrected reduction path
    return MyModel()

def GetInput():
    # Generates a 4D tensor matching the PR's example's semantics
    B, C, H, W = 1, 3, 3, 1  # Matches 3x3 spatial dims in example (reshaped to 4D)
    return torch.randn(B, C, H, W, dtype=torch.float32, device='cuda')

