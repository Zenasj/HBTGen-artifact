import torch.nn as nn

import torch
import torch._dynamo
import torch.func
from torch.fx.experimental import proxy_tensor
from torch._dispatch.python import enable_python_dispatcher

def func(x, y):
    return torch.matmul(x, y)

x = torch.randn(2, 4, 3, 4)
y = torch.randn(2, 4, 4, 3)

with enable_python_dispatcher():
    # RuntimeError: Cannot call sizes() on tensor with symbolic sizes/strides
    gm = proxy_tensor.make_fx(torch.func.functionalize(func), tracing_mode="symbolic")(x, y)

import torch
import torch._dynamo
import torch.func
from torch.fx.experimental import proxy_tensor

def func(x, y):
    return torch.matmul(x, y.transpose(-1, -2))

x = torch.randn(2, 4, 3, 4)
y = torch.randn(2, 4, 3, 4)


gm, _ = torch._dynamo.export(func, x, y)
gm.print_readable()
gm = proxy_tensor.make_fx(torch.func.functionalize(gm), tracing_mode="symbolic")(x, y)
gm.print_readable()

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = torch.tensor([True, False])
    def forward(self, x):
        x.view(3, 2).masked_fill_(self.mask.unsqueeze(0), torch.finfo(x.dtype).max)
        return x

gm, _ = torch._dynamo.export(m, x, tracing_mode="symbolic", aten_graph=True)
gm.print_readable()
"""
class GraphModule(torch.nn.Module):
    def forward(self, x):
        arg0: f32[6], = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
        # File: test_mutation.py:8, code: x.view(3, 2).masked_fill_(self.mask.unsqueeze(0), torch.finfo(x.dtype).max)
        view_default: f32[3, 2] = torch.ops.aten.view.default(arg0, [3, 2])
        _tensor_constant0 = self._tensor_constant0
        unsqueeze_default: b8[1, 2] = torch.ops.aten.unsqueeze.default(_tensor_constant0, 0);  _tensor_constant0 = None
        masked_fill__scalar: f32[3, 2] = torch.ops.aten.masked_fill_.Scalar(view_default, unsqueeze_default, 3.4028234663852886e+38);  view_default = unsqueeze_default = None
        return pytree.tree_unflatten([arg0], self._out_spec)
"""

gm, _ = torch._dynamo.export(func, x, y)
with enable_python_dispatcher():
    gm = proxy_tensor.make_fx(torch.func.functionalize(gm), tracing_mode="symbolic")(x, y)

import torch
import torch._dynamo
import torch.func
from torch.fx.experimental import proxy_tensor
from torch._dispatch.python import enable_python_dispatcher

def func(x, y):
    return torch.matmul(x, y)

x = torch.randn(2, 4, 3, 4)
y = torch.randn(2, 4, 4, 3)

with enable_python_dispatcher():
    # RuntimeError: Cannot call sizes() on tensor with symbolic sizes/strides
    gm = proxy_tensor.make_fx(torch.func.functionalize(func), tracing_mode="symbolic")(x, y)