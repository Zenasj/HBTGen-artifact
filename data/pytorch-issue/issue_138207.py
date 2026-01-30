import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.split_module import split_module
import copy

def foo(x):
    x.add_(1)
    return None

g = make_fx(foo, tracing_mode="fake")(torch.randn(3,))
g.print_readable()
copy.deepcopy(g)  # This works

def cb(node):
    return 1

# sp_gm returns a sub-graph with no output.
sp_gm = split_module(g, None, cb)  
sp_gm.print_readable()
copy.deepcopy(sp_gm)  # This fails