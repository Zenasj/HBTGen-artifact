import torch
import torch.fx
from torch.overrides import wrap_torch_function

@wrap_torch_function(lambda t, x, y: t)
def quantize(t, x, y):
    return t * x + y

def linear_replace_fn(i, w):
    return quantize(i, 5.0, 0)
linear_replace = torch.fx.symbolic_trace(linear_replace_fn)