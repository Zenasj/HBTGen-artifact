import torch

@torch.compile
def indexit(A, b):
    return A[b - 1]

A = torch.rand(20)
b = torch.zeros(4)

print(indexit(A, b))

@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 1
    tmp2 = tmp0 - tmp1
    tmp3 = tl.load(in_ptr1 + (tmp2), xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp3, xmask)