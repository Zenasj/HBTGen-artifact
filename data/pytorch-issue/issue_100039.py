# torch.rand(1, dtype=torch.float32)  # Dummy input tensor (not used in computation)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = torch.tensor(0.001, dtype=torch.float32)  # Default float32 tensor
        self.steps = 1_000_000  # Number of iterations from original example

    def forward(self, _):
        # Accumulate using Python float (64-bit) and torch.float32 tensor
        x = 0.0  # Python float (64-bit)
        y = torch.tensor(0.0, dtype=torch.float32)  # torch.float32 accumulator
        eps_val = self.eps.item()  # Convert to 64-bit float via .item()
        
        for _ in range(self.steps):
            x += eps_val  # 64-bit accumulation
            y += self.eps  # 32-bit accumulation
        
        # Compare final values using torch operations
        y_float = y.float()  # Ensure same precision for comparison
        difference = torch.abs(torch.tensor(x) - y_float)
        return difference  # Return absolute difference as tensor

def my_model_function():
    return MyModel()

def GetInput():
    # Dummy input to satisfy model interface requirements
    return torch.rand(1, dtype=torch.float32)

