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

def torch_square(x):
  x = x[x > 2]
  n_elements = x.numel()
  output = torch.zeros_like(x)
  grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
  square[grid](x, output, n_elements, BLOCK_SIZE=16)
  return output


t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

compiled_func = torch.compile(
    torch_square, backend="aot_eager", dynamic=True
)

print(compiled_func(t))