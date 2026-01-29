# torch.rand(5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compare in-place and non-inplace Bernoulli outputs
        x_inplace = x.clone()
        x_inplace.bernoulli_()
        x_noninplace = x.bernoulli()
        return (x_inplace, x_noninplace)  # Return both outputs for cross-platform comparison

def my_model_function():
    return MyModel()

def GetInput():
    # Replicate the input tensor from the issue (probabilities set to 0.5 for meaningful sampling)
    return torch.full((5,), 0.5, dtype=torch.float32)

