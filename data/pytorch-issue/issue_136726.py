# torch.rand(1, 1)  # Dummy input shape (no actual input used by model)
import torch
import torch.nn as nn

class Test:
    def __init__(self):
        self.count = 0

    def hello(self):
        return None

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.test_sub = Test()  # Submodule containing Test class
        
    def forward(self, x):
        # Simulate RPC flow (note: actual RPC setup requires external handling)
        self.test_sub.hello()
        return x  # Pass-through output for compatibility

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1)  # Dummy input tensor

