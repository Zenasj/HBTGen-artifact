# torch.rand(1, 1, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def compare_tensors(self, expected, actual):
        if expected.shape != actual.shape:
            try:
                expected = expected.expand_as(actual)
            except RuntimeError:
                return False  # Expansion failed
        return torch.allclose(expected, actual)
    
    def forward(self, x):
        # Generate tensors with different shapes
        A = torch.zeros(1, 1, dtype=x.dtype, device=x.device)  # Shape (1,1)
        B = torch.zeros(1, dtype=x.dtype, device=x.device)     # Shape (1)
        
        # Compare in both directions (A vs B and B vs A)
        res1 = self.compare_tensors(A, B)  # Expected=A, Actual=B
        res2 = self.compare_tensors(B, A)  # Expected=B, Actual=A
        return torch.tensor([res1, res2], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

