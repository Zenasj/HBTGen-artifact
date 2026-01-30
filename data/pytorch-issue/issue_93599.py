import torch
import torch._dynamo
import torch._inductor
from torch._inductor import config
import logging
from torchvision import models

resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

batch_size = 4096
device = "cuda"

resnet18 = resnet18.eval().to(device)
opt_resnet18 = torch._dynamo.optimize("inductor")(resnet18)

input = torch.randn((batch_size, 3, 224, 224)).to(device)
output = opt_resnet18(input)
print(output.shape)

from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


kernel0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4294967296], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'u32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3288334336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp11 = tl.load(in_ptr3 + (x1), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
''')




async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_63, primals_64, primals_123 = args
    args.clear()
    buf0 = aten.convolution(primals_123, primals_1, None, (2, 2), (3, 3), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf0, (4096, 64, 112, 112), (802816, 12544, 112, 1))
    buf1 = empty_strided((4096, 64, 112, 112), (802816, 12544, 112, 1), device='cuda', dtype=torch.float32)
    stream0 = get_cuda_stream(0)
    kernel0.run(buf0, primals_63, primals_64, primals_2, primals_3, buf1, 3288334336, grid=grid(3288334336), stream=stream0)
    return (buf1,)


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
    primals_64 = rand_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
    primals_123 = rand_strided((4096, 3, 224, 224), (150528, 50176, 224, 1), device='cuda', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_2, primals_3,  primals_63, primals_64,  primals_123]))