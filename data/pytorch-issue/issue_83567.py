from torch.fx.experimental.proxy_tensor import make_fx
import torch

def f(x, y):
    return x[0, y:]