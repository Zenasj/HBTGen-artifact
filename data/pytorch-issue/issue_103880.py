import torch

def fn(a):
    result = torch.deg2rad(a).sin()
    return torch.empty((128, 128), device=a.device).fill_(result)

@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset  + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = 0.017453292519943295
    tmp3 = tmp1 * tmp2
    tmp4 = tl.sin(tmp3)
    tl.store(out_ptr0 + (x0), tmp4, None)