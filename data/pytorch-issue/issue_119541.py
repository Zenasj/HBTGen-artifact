import torch
from torch._higher_order_ops.triton_kernel_wrap import (
    kernel_side_table,
    triton_kernel_wrapper_functional,
    triton_kernel_wrapper_mutation,
)
from functorch import make_fx

import triton
from triton import language as tl

@triton.jit
def add(
    in_ptr0,
    in_ptr1,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = y + x
    tl.store(out_ptr + offsets, output, mask=mask)

kernel_side_table.reset_table()

def f(x, y, output):
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add[grid](x, y, output, n_elements, BLOCK_SIZE=16)
    return output


t1 = torch.rand(5, device="cuda")
t2 = torch.rand(5, device="cuda")
out = torch.rand(5, device="cuda")

compiled_func = torch.compile(
    f, backend="inductor", fullgraph=True, dynamic=False
)

print(f(t1, t2, out))

print(compiled_func(t1, t2, out))

import torch
from torch._higher_order_ops.triton_kernel_wrap import (
    kernel_side_table,
    triton_kernel_wrapper_functional,
    triton_kernel_wrapper_mutation,
)
from functorch import make_fx

import triton
from triton import language as tl

@triton.jit
def add(
    in_ptr0,
    in_ptr1,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = y + x
    tl.store(out_ptr + offsets, output, mask=mask)

kernel_side_table.reset_table()

def f(x, y, output):
    out = triton_kernel_wrapper_functional(
        kernel_idx=kernel_side_table.add_kernel(add),
        grid=[(x.numel(),)],
        kwargs={
            "in_ptr0": x,
            "in_ptr1": y,
            "out_ptr": output,
            "n_elements": output.numel(),
            "BLOCK_SIZE": 16,
        },
        tensors_to_clone=["out_ptr"],
    )
    return out["out_ptr"]


t1 = torch.rand(5, device="cuda")
t2 = torch.rand(5, device="cuda")
out = torch.rand(5, device="cuda")


gm = make_fx(f, tracing_mode="fake")(t1, t2, out)
print(gm.code.strip())


compiled_func = torch.compile(
    f, backend="inductor", fullgraph=True, dynamic=False
)

print(f(t1, t2, out))

print(compiled_func(t1, t2, out))

import torch
from torch._higher_order_ops.triton_kernel_wrap import (
    kernel_side_table,
    triton_kernel_wrapper_functional,
    triton_kernel_wrapper_mutation,
)
from functorch import make_fx

import triton
from triton import language as tl

@triton.jit
def add(
    in_ptr0,
    in_ptr1,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = y + x
    tl.store(out_ptr + offsets, output, mask=mask)

kernel_side_table.reset_table()

def f(x, y, output):
    triton_kernel_wrapper_mutation(
        kernel_idx=kernel_side_table.add_kernel(add),
        grid=[(x.numel(),)],
        kwargs={
            "in_ptr0": x,
            "in_ptr1": y,
            "out_ptr": output,
            "n_elements": output.numel(),
            "BLOCK_SIZE": 16,
        },
    )
    return output


t1 = torch.rand(5, device="cuda")
t2 = torch.rand(5, device="cuda")
out = torch.rand(5, device="cuda")


gm = make_fx(f, tracing_mode="fake")(t1, t2, out)
print(gm.code.strip())


compiled_func = torch.compile(
    f, backend="inductor", fullgraph=True, dynamic=False
)

print(f(t1, t2, out))

print(compiled_func(t1, t2, out))

tensor([1.4726, 1.1107, 0.7445, 0.7926, 0.7915], device='cuda:0')
tensor([1.4726, 1.1107, 0.7445, 0.7926, 0.7915], device='cuda:0')

import torch
import triton
from triton import language as tl

@triton.jit
def square(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements
  x = tl.load(in_ptr + offsets, mask=mask)
  output = x * x
  tl.store(out_ptr + offsets, output, mask=mask)

def f(x):
  #x = torch.unique(x)
  x = x[x > 2]
  n_elements = x.numel()
  output = torch.zeros_like(x, device="cuda")
  grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
  square[grid](x, output, n_elements, BLOCK_SIZE=16)
  return output


t = torch.tensor([1, 2, 3, 2, 3, 1, 4], device="cuda")

torch._dynamo.config.capture_dynamic_output_shape_ops = True

compiled_func = torch.compile(
    f, backend="inductor", dynamic=True
)

print(f(t))
print(compiled_func(t))