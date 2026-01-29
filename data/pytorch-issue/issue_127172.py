# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyClass:
    foo: int = 1

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = MyClass()  # Case 1: MyClass instance
        self.m2 = 0          # Case 2: integer 0

    def forward(self, x):
        # Compute outputs for both cases and return difference
        # Case 1: m is MyClass instance (triggers Dynamo error when compiled)
        m = self.m1
        if getattr(type(m), "foo", 0):
            out1 = x + MyClass.foo
        else:
            out1 = x

        # Case 2: m is integer 0 (works without error)
        m = self.m2
        if getattr(type(m), "foo", 0):
            out2 = x + MyClass.foo
        else:
            out2 = x

        # Return tuple indicating outputs (error occurs in case1 when compiled)
        return (out1, out2)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)

