import torch

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (2000, 2000), (2016, 1)) 
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2000, 2000), (2016, 1), torch.float32)
    ...

...

@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2000
    x1 = (xindex // 2000)
    tmp0 = tl.load(in_ptr0 + (x0 + (2016*x1)), xmask)
    tl.store(out_ptr0 + (x0 + (2016*x1)), tmp0, xmask)