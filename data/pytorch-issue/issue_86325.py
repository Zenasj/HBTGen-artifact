# torch.rand(1, dtype=torch.float32)  # Inferred input shape from the provided code

import torch
import torch.nn as nn

class Inner(nn.Module):
    def forward(self, x):
        if x > 0:
            return x
        else:
            return x * x

class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = torch.jit.script(Inner())

    def forward(self, x):
        return self.inner(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.outer = Outer()

    def forward(self, x):
        return self.outer(x)

def my_model_function():
    model = MyModel()
    model.eval()
    traced_model = torch.jit.trace_module(model, {'forward': (torch.zeros(1, dtype=torch.float32),)})
    optimized_model = torch.jit.optimize_for_inference(torch.jit.freeze(traced_model))
    # Workaround for the issue: explicitly set the training attribute
    optimized_model.training = False
    return optimized_model

def GetInput():
    return torch.zeros(1, dtype=torch.float32)

