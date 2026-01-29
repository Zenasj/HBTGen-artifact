import torch
import torch.autograd.forward_ad as fwAD
from torch import nn

# torch.rand(3, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 4312491 * x  # Scalar multiplication triggering the wrapped number issue

def my_model_function():
    return MyModel()

def GetInput():
    device = "cpu"
    with fwAD.dual_level():
        x = torch.randn(3, device=device)  # Base tensor of shape (3,)
        y = torch.ones_like(x)            # Tangent component for forward AD
        dual = fwAD.make_dual(x, y)       # Creates dual tensor (input to MyModel)
        return dual

