# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for _ in range(10)])
        self._ml_len = len(self.linears)  # Precompute length to avoid using len() in forward

    def forward(self, x):
        # Process input through all linear layers and scale by the precomputed length
        for linear in self.linears:
            x = linear(x)
        return x * self._ml_len  # Use stored length instead of dynamic len()

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input tensor matching the expected shape (B=1, C=10)
    return torch.rand(1, 10, dtype=torch.float32)

