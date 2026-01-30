import torch

x = torch.randn((5,), dtype=torch.half).cuda()
y = torch.randn((5,), dtype=torch.half).cuda()

print(torch.logaddexp(x, y))

import torch

x = torch.randn((5,), dtype=torch.half).cuda()
y = torch.randn((5,), dtype=torch.half).cuda()

@torch.compile
def logaddexp(x, y):
    x = torch.exp(x)
    y = torch.exp(y)
    return torch.log(x + y)

logaddexp(x, y)

@pointwise(size_hints=[8], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': set(), 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp1 = tl.exp(tmp0)
    tmp3 = tl.exp(tmp2)
    tmp4 = tmp1 + tmp3
    tmp5 = tl.log(tmp4)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp5, xmask)

@torch.compile
def logaddexp(x, y):
    m = torch.maximum(x, y).detach()

    return ((x - m).exp() + (y - m).exp()).log() + m