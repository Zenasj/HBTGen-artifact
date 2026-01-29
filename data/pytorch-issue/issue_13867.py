# torch.rand(N, dtype=torch.float, device='cuda')  # N is the number of categories
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, weights):
        # Perform multinomial sampling with replacement and 4096 samples (as per original issue)
        return torch.multinomial(weights, 4096, replacement=True)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a 1D tensor with some zeros and small positive values on CUDA
    input_size = 1000  # Matches original issue's "weights.shape[0]"
    weights = torch.rand(input_size, device='cuda') * 0.0001  # Create small values
    zeros_mask = torch.rand(input_size, device='cuda') < 0.5  # 50% chance to be zero
    weights[zeros_mask] = 0.0
    # Ensure at least one non-zero element to avoid zero-sum weights
    if weights.sum() == 0:
        weights[0] = 1e-5
    return weights

