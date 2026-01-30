import torch.library
from typing import Tuple

@torch.library.custom_op(
    "mylib::index_op", mutates_args=(), device_types=["cpu"]
)
def index_op(indices: Tensor, values: Tensor) -> Tuple[int, Tensor]:
    u0 = indices.max().item()
    return u0, (values + u0).clone()

def index_op_backward(ctx, _grad_u0, grad):
   u0 = ctx.u0
   return None, grad * u0

def index_op_setup_context(ctx, inputs, output):
   indices, values = inputs
   u0, out = output
   ctx.u0 = u0
   # ctx.save_for_backward(u0)

torch.library.register_autograd(
    "mylib::index_op", index_op_backward, setup_context=index_op_setup_context)

@torch.library.register_fake("mylib::index_op")
def _(indices: Tensor, values: Tensor):
    u0 = torch.library.get_ctx().new_dynamic_size()
    return u0, torch.empty_like(values)

@torch.compile(backend="inductor")
def cf_index_op(indices, values):
    return index_op(indices, values)

@run_test
def test_index_op():
    x = torch.randn(20, requires_grad=True)
    r = cf_index_op(torch.tensor([0, 1, 2]), x)
    r[1].sum().backward()

import torch.library
from typing import Tuple

@torch.library.custom_op(
    "mylib::index_op", mutates_args=(), device_types=["cpu"]
)
def index_op(indices: Tensor, values: Tensor) -> Tuple[Tensor, Tensor]:
    t0 = indices.max()
    u0 = t0.item()
    return t0, (values + u0).clone()

def index_op_backward(ctx, _grad_u0, grad):
   t0, = ctx.saved_tensors
   u0 = t0.item()
   return None, grad * u0

def index_op_setup_context(ctx, inputs, output):
   indices, values = inputs
   t0, out = output
   ctx.save_for_backward(t0)

torch.library.register_autograd(
    "mylib::index_op", index_op_backward, setup_context=index_op_setup_context)

@torch.library.register_fake("mylib::index_op")
def _(indices: Tensor, values: Tensor):
    # u0 = torch.library.get_ctx().new_dynamic_size()
    return torch.empty([1], dtype=torch.int64), torch.empty_like(values)

@torch.compile(backend="inductor")
def cf_index_op(indices, values):
    return index_op(indices, values)

@run_test
def test_index_op():
    x = torch.randn(20, requires_grad=True)
    r = cf_index_op(torch.tensor([0, 1, 2]), x)
    r[1].sum().backward()

import torch.library
from typing import Tuple

@torch.library.custom_op(
    "mylib::index_op", mutates_args=(), device_types=["cpu"]
)
def index_op(indices: Tensor, values: Tensor) -> Tuple[Tensor, Tensor]:
    t0 = indices.max()
    u0 = t0.item()
    return (values + u0).clone()

def index_op_backward(ctx, _grad_u0, grad):
   u0 = ctx.u0
   return None, grad * u0

def index_op_setup_context(ctx, inputs, output):
   indices, values = inputs
   t0, out = output
   u0 = t0.item()
   ctx.u0 = u0

torch.library.register_autograd(
    "mylib::index_op", index_op_backward, setup_context=index_op_setup_context)

@torch.library.register_fake("mylib::index_op")
def _(indices: Tensor, values: Tensor):
    # u0 = torch.library.get_ctx().new_dynamic_size()
    return torch.empty([1], dtype=torch.int64), torch.empty_like(values)

@torch.compile(backend="inductor")
def cf_index_op(indices, values):
    return index_op(indices, values)

@run_test
def test_index_op():
    x = torch.randn(20, requires_grad=True)
    r = cf_index_op(torch.tensor([0, 1, 2]), x)
    r[1].sum().backward()

import torch.library
from typing import Tuple

@torch.library.custom_op(
    "mylib::index_op", mutates_args=(), device_types=["cpu"]
)
def index_op(indices: Tensor, values: Tensor) -> Tensor:
    t0 = indices.max()
    u0 = t0.item()
    return (values + u0).clone()

def index_op_backward(ctx, grad):
   t0, = ctx.saved_tensors
   u0 = t0.item()
   return None, grad * u0

def index_op_setup_context(ctx, inputs, output):
   indices, values = inputs
   out = output
   t0 = indices.max()
   ctx.save_for_backward(t0)

torch.library.register_autograd(
    "mylib::index_op", index_op_backward, setup_context=index_op_setup_context)

@torch.library.register_fake("mylib::index_op")
def _(indices: Tensor, values: Tensor):
    return torch.empty_like(values)

@torch.compile(backend="inductor")
def cf_index_op(indices, values):
    return index_op(indices, values)

@run_test
def test_index_op():
    x = torch.randn(20, requires_grad=True)
    r = cf_index_op(torch.tensor([0, 1, 2]), x)
    r[1].sum().backward()