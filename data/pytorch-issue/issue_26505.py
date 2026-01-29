# torch.rand(1, 4, dtype=torch.float32)
import torch
import torch.nn as nn

@torch.jit.script
def f(y):
    # type: (Tuple[str, str, str, str])
    return list(y)

class MyModel(nn.Module):
    def forward(self, x):
        # Trigger the JIT error by calling the problematic function
        t = ("a", "b", "c", "d")
        result = f(t)  # This line will cause the VarType list matching error in TorchScript
        return x  # Return input tensor as-is (not used in error path)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 4, dtype=torch.float32)

