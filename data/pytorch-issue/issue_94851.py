# torch.rand(13, 192, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
from torch import nn
from torch.func import grad, functional_call
from torch._dynamo import allow_in_graph
from functools import wraps

class MyModel(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.W = nn.Parameter(torch.empty((int(4 * dim), dim)))
        nn.init.normal_(self.W)

    def forward(self, x):
        hid = torch.einsum("...d,yd->...y", x, self.W)
        return hid.sum()

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn((13, 192))

def traceable(f):
    f = allow_in_graph(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper

def functionalized_forward(model, x):
    params = {name: param for name, param in model.named_parameters()}
    buffers = {name: buffer for name, buffer in model.named_buffers()}
    return functional_call(model, (params, buffers), (x,))

def compiled_grad(model, x):
    f = lambda x: functionalized_forward(model, x)
    fjit = torch.compile(traceable(grad(f)))
    return fjit(x)

# Example usage:
# model = my_model_function()
# x = GetInput()
# result = compiled_grad(model, x)

