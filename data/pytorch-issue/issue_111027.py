import torch

def fn(a):
    b = a.view((2, 2))
    return b.sgn()

x_cpu =torch.tensor([[2.0, 2], [-2, -2]], requires_grad=True)
compiled_fn = torch.compile(fn)        
y_cpu = compiled_fn(x_cpu)
print("y_hpu", y_cpu)

import torch

def f():
    torch._enable_functionalization(reapply_views=True)
    x = torch._efficientzerotensor(4)
    y = x.reshape(-1)
    torch._sync(y)
    return y

out = f()

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._dispatch.python import enable_python_dispatcher

def fn(a):
    b = torch.mul(a, 2)
    out = b.sgn()
    a_grad = torch.autograd.grad([out], [a], grad_outputs=[torch.ones_like(out)])
    return a_grad

x = torch.ones(2, 2, requires_grad=True, dtype=torch.float32)
with enable_python_dispatcher():
    fx_g = make_fx(fn, decomposition_table=torch._decomp.core_aten_decompositions())(x)
print(fx_g.code)

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._dispatch.python import enable_python_dispatcher

def fn(a):
    b = torch.add(a, a)
    out = b.sgn()
    a_grad = torch.autograd.grad([out], [a], grad_outputs=[torch.ones_like(out)])
    return a_grad

x = torch.ones(2, 2, requires_grad=True, dtype=torch.float32)
with enable_python_dispatcher():
    fx_g = make_fx(fn, decomposition_table=torch._decomp.core_aten_decompositions())(x)
print(fx_g.code)

def forward(self, a_1):

    # forward
    add = torch.ops.aten.add.Tensor(a_1, a_1);  a_1 = None
    sign = torch.ops.aten.sign.default(add);  add = None
    
    # backward
    alias = torch.ops.aten.alias.default(sign)
    full_like = torch.ops.aten.full_like.default(sign, 1, pin_memory = False, memory_format = torch.preserve_format)
    is_same_size = torch.ops.aten.is_same_size.default(sign, full_like);  sign = full_like = None
    alias_1 = torch.ops.aten.alias.default(alias);  alias = None
    _efficientzerotensor = torch.ops.aten._efficientzerotensor.default([2, 2], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    _to_copy = torch.ops.aten._to_copy.default(_efficientzerotensor, device = device(type='meta'))
    _to_copy_1 = torch.ops.aten._to_copy.default(_efficientzerotensor, device = device(type='meta'));  _efficientzerotensor = None
    add_1 = torch.ops.prims.add.default(_to_copy_1, _to_copy);  _to_copy_1 = _to_copy = None
    _efficientzerotensor_1 = torch.ops.aten._efficientzerotensor.default([2, 2], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    return (_efficientzerotensor_1,)