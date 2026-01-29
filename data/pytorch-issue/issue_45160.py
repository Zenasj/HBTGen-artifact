# torch.rand(1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Replicates the operations in the original issue's udf_with_torch_ops
        t = torch.add(x, x)  # Equivalent to t1 + t2 where t1=t2=torch.ones(1)
        t = torch.mul(t, t)  # (2)^2 = 4
        t = t.relu()         # No change since value is positive
        t = t.sigmoid()      # Final activation
        return t

def my_model_function():
    # Returns a model instance with default initialization
    return MyModel()

def GetInput():
    # Generates a random input matching the expected shape (1-element tensor)
    return torch.rand(1)

