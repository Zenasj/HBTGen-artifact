import triton
import triton.language as tl

@triton.autotune( # E: Untyped decorator makes function "sin_kernel" untyped  [misc]
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=4),
    ],
    key=['n_elements']
)
@triton.jit # E: Untyped decorator makes function "sin_kernel" untyped  [misc]
def sin_kernel( # E: Function is missing a return type annotation  [no-untyped-def]
    in_ptr0,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    if in_ptr0 is not None:
        x = tl.load(in_ptr0 + offsets, mask=mask)
    else:
        x = 0.
    output = tl.sin(x)
    tl.store(out_ptr + offsets, output, mask=mask)

import torch

def sin_triton(x, out):
    n_elements = out.numel()
    sin_kernel[(n_elements,)](x, out, n_elements)

x = torch.randn(65, device="cuda")
out = torch.empty_like(x)
out_compiled = torch.empty_like(x)

sin_triton_compiled = torch.compile(fullgraph=True)(sin_triton)

for first in (x, None):
    sin_triton(first, out)
    sin_triton_compiled(first, out_compiled)
    torch.testing.assert_close(out, out_compiled)