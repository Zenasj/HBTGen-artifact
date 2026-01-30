import torch
from torch.fx.experimental.proxy_tensor import make_fx

decompositions = {
    torch.ops.aten.max_pool2d.default: lambda x, *_: x
}

def fwd(x):
    return torch.max_pool2d(x, (1, 1))

with torch.inference_mode():
    # fwd = torch.func.functionalize(fwd)
    fx = make_fx(fwd, decompositions)(torch.randn(1, 1, 1, 1))
fx.graph.print_tabular()