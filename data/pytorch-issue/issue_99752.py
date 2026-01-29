# torch.rand((), dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compare two methods of converting a tensor to a bool
        method1 = bool(x)  # Python's bool() on the tensor
        method2 = torch.ops.aten.Bool(x.item())  # aten.Bool op on the numeric value
        # Return tensor indicating if the two methods agree
        return torch.tensor(int(method1 == method2))

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a scalar float tensor (0-dim) to test bool conversion
    return torch.rand(())

