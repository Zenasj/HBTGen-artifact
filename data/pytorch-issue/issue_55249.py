# torch.randint(0, 10, (3,), dtype=torch.int32)  # Input shape: 1D tensor of 3 integers
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Extract tuples from input tensor
        a = (x[0].item(), x[1].item())  # Original a: Tuple[int, int]
        b = (x[2].item(),)              # Original b: Tuple[int]
        
        # Python-style tuple concatenation (expected behavior)
        py_result = a + b
        
        # Simulate TorchScript behavior (returns list instead of tuple)
        ts_result = list(a) + list(b)    # Convert tuples to lists and concatenate
        
        # Return 0 if types differ (tuple vs list), 1 if same (should never happen)
        return torch.tensor(0 if type(py_result) != type(ts_result) else 1)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random 1D tensor of 3 integers (matches input shape expectation)
    return torch.randint(0, 10, (3,), dtype=torch.int32)

