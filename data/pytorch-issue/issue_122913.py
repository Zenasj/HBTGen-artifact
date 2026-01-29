# torch.rand(B, C, H, W, dtype=torch.float32)  # Example: B=2, C=3, H=1 (constrained singleton), W=5
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Symbolic dimension constraint check (H must be singleton)
        h = x.shape[2]
        if h != 1:
            raise ValueError("H dimension must be constrained to singleton")
        return x.view(x.size(0), -1)  # Force shape dependency for Dynamo testing

def my_model_function():
    return MyModel()

def GetInput():
    # Valid input with constrained H=1 (matches PR's singleton constraint scenario)
    return torch.rand(2, 3, 1, 5, dtype=torch.float32)

