import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.split_module import split_module

def fn(x):
    return (x,)

g = make_fx(fn, tracing_mode="fake")(torch.randn(3, 3))

g.print_readable()

# `keep_original_order=False` works
# split_module(g, None, split_callback=lambda _ : 0, keep_original_order=False)

# This fails
split_module(g, None, split_callback=lambda _ : 0, keep_original_order=True)