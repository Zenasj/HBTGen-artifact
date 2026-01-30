import torch
import torch.nn as nn

@torch.jit.script
def bad_error(x):
    return torch.nn.functional.interpolate(x, 'bad')