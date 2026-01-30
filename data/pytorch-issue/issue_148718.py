import torch
import torch._inductor.config as config

config.triton.prefer_nd_tiling = True
config.triton.use_block_ptr = True

full_size = (21, 32)
view_size = (21, 19)
def get_input():
    full = torch.rand(full_size, device="cuda")
    view = torch.as_strided(full, view_size, full.stride())
    return view

inps = [get_input() for _ in range(2)]

compiled = torch.compile(torch.add)
compiled(*inps)

@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 21
    xnumel = 19
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 32*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1 + 32*y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x1 + 19*y0), tmp2, xmask & ymask)

xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
yindex = yoffset + tl.arange(0, YBLOCK)[None, :]

xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
yindex = yoffset + tl.arange(0, YBLOCK)[:, None]

@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    combined_idx = tl.reshape(x1 + 5 * y0, [XBLOCK * YBLOCK]) # Debugging
    tl.device_print("linear_idx", combined_idx)
    tmp0 = tl.load(in_ptr0 + (x1 + 5*y0), xmask & ymask)
    tmp1 = tl.load(in_ptr1 + (x1 + 5*y0), xmask & ymask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x1 + 4*y0), tmp2, xmask & ymask)

@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    combined_idx = tl.reshape(x1 + 5 * y0, [XBLOCK * YBLOCK]) # Debugging
    tl.device_print("linear_idx", combined_idx)
    tmp0 = tl.load(in_ptr0 + (x1 + 5*y0), xmask & ymask)
    tmp1 = tl.load(in_ptr1 + (x1 + 5*y0), xmask & ymask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x1 + 4*y0), tmp2, xmask & ymask)

@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    combined_idx = tl.reshape(y0 + 5 * x1, [XBLOCK * YBLOCK]) # Debugging
    tl.device_print("linear_idx", combined_idx)
    tmp0 = tl.load(in_ptr0 + (y0 + 5*x1), xmask & ymask)
    tmp1 = tl.load(in_ptr1 + (y0 + 5*x1), xmask & ymask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + 4*x1), tmp2, xmask & ymask)