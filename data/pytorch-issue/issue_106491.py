import torch
import torch.nn as nn

# torch.rand(3, dtype=torch.float32)  # Inferred input shape from test case
class MyModel(nn.Module):
    def forward(self, x):
        def outer(a):
            return a + 1

        def direct(x):
            def deep2(c):
                return outer(c)
            def deep(c):
                return deep2(c)
            return deep(x)
        
        return direct(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3)

