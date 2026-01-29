# torch.rand(0, 3, 4, 5, dtype=torch.float32)  # Input shape with empty batch dimension to test empty shard
import torch
from torch import nn

class BuggyGather(nn.Module):
    def forward(self, x):
        # Simulates original gather logic that would crash on empty shards
        # Placeholder: return invalid tensor if empty (e.g., KeyError-like behavior)
        if x.numel() == 0:
            return torch.tensor(float('nan'))  # Simulate error by returning invalid value
        return x + 1  # Mock gather operation

class FixedGather(nn.Module):
    def forward(self, x):
        # Fixed logic with empty check
        if x.numel() == 0:
            return torch.empty(0, *x.shape[1:], dtype=x.dtype)  # Return empty tensor
        return x + 1  # Same mock operation as before

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.buggy = BuggyGather()
        self.fixed = FixedGather()
        
    def forward(self, x):
        # Compare outputs of both implementations
        buggy_out = self.buggy(x)
        fixed_out = self.fixed(x)
        # Use isfinite to detect invalid outputs (simulating error handling)
        return torch.allclose(buggy_out, fixed_out) and torch.all(torch.isfinite(buggy_out))

def my_model_function():
    return MyModel()

def GetInput():
    # Creates input with empty batch dimension (simulating empty shard)
    return torch.empty(0, 3, 4, 5, dtype=torch.float32)

