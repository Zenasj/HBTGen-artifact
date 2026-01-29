# torch.rand(3, dtype=torch.float32)  # Input shape inferred from example
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Hardcode the example string since inputs must be tensors
        self.code_str = "print('Hello World')"  # Matches the example in the PR
        
    def forward(self, x):
        # Replicate the Dynamo compilation scenario from the PR example
        code = compile(self.code_str, "foo", "exec")
        exec(code)
        return x  # Return the input tensor as in the original function

def my_model_function():
    # Return model instance with fixed code_str for reproducibility
    return MyModel()

def GetInput():
    # Return input matching the example's torch.rand(3)
    return torch.rand(3, dtype=torch.float32)

