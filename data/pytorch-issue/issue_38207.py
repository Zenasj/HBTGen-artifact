# torch.rand(1)  # Dummy input tensor (not used in processing)
import torch
from torch import nn
from typing import List

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the Python split function and its JIT version
        def simple_split(txt: str) -> List[str]:
            return txt.split()
        self.py_split = simple_split
        self.jit_split = torch.jit.script(simple_split)
    
    def forward(self, x):
        # Use the example string from the original issue
        test_str = 'simple     split example'
        py_result = self.py_split(test_str)
        jit_result = self.jit_split(test_str)
        # Return True if outputs match, else False (as a tensor)
        return torch.tensor(py_result == jit_result, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a dummy tensor (shape and dtype are arbitrary since input isn't used)
    return torch.rand(1)

