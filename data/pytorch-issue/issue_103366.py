# torch.rand(1, dtype=torch.int32), torch.rand(1, dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        x1, x2 = inputs
        try:
            original = torch.inner(x1, x2)
            original_ok = True
        except RuntimeError:
            original = None
            original_ok = False
        fixed = self.fixed_inner(x1, x2)
        # Return comparison result only if original didn't error and dtypes match
        if original_ok and fixed.dtype == original.dtype:
            return torch.allclose(original, fixed)
        else:
            return torch.tensor(False)  # Return tensor for torch.compile compatibility
    
    def fixed_inner(self, x1, x2):
        # Compute inner product manually with dtype preservation
        product = x1 * x2
        return product.sum(-1, dtype=x1.dtype)

def my_model_function():
    return MyModel()

def GetInput():
    # Create two 1-element tensors with dtype int32
    x1 = torch.randint(0, 10, (1,), dtype=torch.int32)
    x2 = torch.randint(0, 10, (1,), dtype=torch.int32)
    return (x1, x2)

