# torch.rand(10, 2, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        noncontig = x.movedim(-1, 0)
        expected = torch.narrow_copy(noncontig.contiguous(), 1, 0, 10)
        actual = torch.narrow_copy(noncontig, 1, 0, 10)
        # Comparison with tolerances inferred from error message
        is_close = torch.allclose(actual, expected, atol=1e-5, rtol=1.3e-6)
        return torch.tensor(is_close, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(10, 2, dtype=torch.float)

