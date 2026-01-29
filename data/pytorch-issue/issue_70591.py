# (torch.rand((), dtype=torch.bool), torch.rand((), dtype=torch.int64), torch.rand((), dtype=torch.float32))
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        results = []
        for x in inputs:
            scalar = x.item()
            # Compare current asarray behavior vs expected (as_tensor)
            arr = torch.asarray(scalar)
            arr2 = torch.as_tensor(scalar)  # Correct dtype reference
            results.append(arr.dtype == arr2.dtype)
        return torch.tensor(results, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return (
        torch.tensor(True),
        torch.tensor(1),
        torch.tensor(1.0),
    )

