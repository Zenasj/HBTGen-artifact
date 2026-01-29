# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

@torch.jit.script
def smooth_l1(x, beta: float):
    t = x.abs()
    return torch.where(x < beta, 0.5 * t ** 2 / beta, t - 0.5 * beta)

class MyModel(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta  # Hyperparameter for smooth L1 function

    def forward(self, x):
        return smooth_l1(x, self.beta)

def my_model_function():
    # Returns model instance with default beta=1.0 as in original code
    return MyModel(beta=1.0)

def GetInput():
    # Generate 4D tensor matching common image input dimensions
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)

