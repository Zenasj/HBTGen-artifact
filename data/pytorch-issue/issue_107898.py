# torch.rand((), dtype=torch.float32)  # Input is a scalar tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Create a tensor with value 32.0 (scalar) and same dtype as input
        a = torch.tensor(32., dtype=x.dtype)
        # Extract scalar value via .item(), which triggers the discussed error under FakeTensorMode + inference_mode
        scalar = a.item()
        return x + scalar  # Returns tensor (x is tensor, scalar is Python number)

def my_model_function():
    # Returns the model instance
    return MyModel()

def GetInput():
    # Returns a scalar tensor as input (matches model's expected input)
    return torch.rand((), dtype=torch.float32)

