import torch.nn as nn

import torch
from torch.library import triton_op, wrap_triton
import triton
from triton import language as tl
@triton.jit
def add_kernel(
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
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

@triton_op("mylib::add", mutates_args={})
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # NB: we need to wrap the triton kernel in a call to wrap_triton
    wrap_triton(add_kernel)[grid](x, y, output, n_elements, 16)
    return output
@torch.compile
def f(x, y):
    return add(x, y)
x = torch.randn(3, device="cuda")
y = torch.randn(3, device="cuda")
z = f(x, y)
assert torch.allclose(z, x + y)
with torch.no_grad():
    torch.onnx.export(f,
                      (x,y,),
                      "triton_export.onnx",  
                      export_params=True,  
                      dynamo=True,
                      opset_version=18,  
                      do_constant_folding=False, 
                      optimize=False,
                      #custom_translation_table=custom_translation_table,
                      input_names=["zzq_a","zzq_b"],
                      output_names=["zzq_out"],
                      verbose=True)

import torch
from torch.library import triton_op, wrap_triton
import triton
from triton import language as tl
import onnxscript
from onnxscript import opset18


@triton.jit
def add_kernel(
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
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)


@triton_op("mylib::test_add", mutates_args={})
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # NB: we need to wrap the triton kernel in a call to wrap_triton
    wrap_triton(add_kernel)[grid](x, y, output, n_elements, 16)
    return output


@torch.compile
def f(x, y):
    return add(x, y)

@onnxscript.script()
def test_add(input1, input2):
    # fake compution
    return opset18.test_add(input1, input2)


onnx_registry = torch.onnx.OnnxRegistry()
onnx_registry.register_op(
    namespace="mylib", op_name="test_add", overload="default", function=test_add)


x = torch.randn(3, device="cuda")
y = torch.randn(3, device="cuda")
z = f(x, y)
assert torch.allclose(z, x + y)
export_options = torch.onnx.ExportOptions(
	dynamic_shapes=True, onnx_registry=onnx_registry)

with torch.no_grad():
	onnx_program = torch.onnx.dynamo_export(f,
                                         x,y, export_options=export_options)
onnx_program.save("triton_dynamo_export.onnx")

import torch
from torch.library import triton_op, wrap_triton
import triton
from triton import language as tl
import onnxscript
from onnxscript import opset18


@triton.jit
def add_kernel(
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
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)


@triton_op("mylib::test_add", mutates_args={})
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # NB: we need to wrap the triton kernel in a call to wrap_triton
    wrap_triton(add_kernel)[grid](x, y, output, n_elements, 16)
    return output


class M(torch.nn.Module):
    def forward(self, x, y):
        return add(x, y) + x


@onnxscript.script()
def test_add(input1, input2):
    # fake compution
    return opset18.test_add(input1, input2)


x = torch.randn(3, device="cuda")
y = torch.randn(3, device="cuda")
f = M()
z = f(x, y)
assert torch.allclose(z, x + y +x)

with torch.no_grad():
    ep = torch.export.export(f, (x, y))
    print(ep)

print("-------------------export------------")
onnx_program = torch.onnx.export(
    f,
    (x, y),
    "triton_export_dynamo_true.onnx",
    dynamo=True,
    custom_translation_table={
        torch.ops.mylib.test_add.default: test_add,
    },
    optimize=False
)

print(onnx_program)