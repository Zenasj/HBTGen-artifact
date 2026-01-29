# torch.rand(2, dtype=torch.float32)  # Input shape inferred as 1D tensor of size 2
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_model = self.CondModel()

    class CondModel(nn.Module):
        def forward(self, x):
            z = torch.ones_like(x)

            def true_fn(x, z):
                x = x + 1.0
                z = z * 1.0  # Placeholder operation to match output structure
                return x, z

            def false_fn(x, z):
                x = x - 1.0
                z = z * 0.0  # Placeholder operation to match output structure
                return x, z

            # Use tuple return from cond, which is now supported
            return torch.cond(x.sum() > 0, true_fn, false_fn, (x, z))

    def forward(self, x):
        return self.cond_model(x)

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a random tensor matching the input shape (2 elements)
    return torch.rand(2, dtype=torch.float32)

