import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as t

class UNet(nn.Module):
    def __init__(self, out_size: int = 32):
        super().__init__()

        self.c2 = nn.Conv2d(out_size, 4 * out_size, 4, stride=4)
        self.u2 = nn.ConvTranspose2d(4 * out_size, out_size, 4, stride=4)

        self.merge = nn.Conv2d(2 * out_size, out_size, 1)

    def forward(self, X):
        Z = self.u2(self.c2(X))
        return self.merge(torch.cat([X, Z], dim=1))

model = UNet(out_size=32)
model.cuda()

model = torch.compile(model, backend="inductor")

out = model(torch.randn(192,32,256,256).cuda())
out.sum().backward()
print("done1", out.shape)

model.zero_grad()

out = model(torch.randn(99,32,256,256).cuda())
out.sum().backward()
print("done2", out.shape)

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config





isolate_fails_code_str = None



# torch version: 2.1.1+cu121
# torch cuda version: 12.1
# torch git version: 4c55dc50355d5e923642c59ad2a23d6ad54711e7


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# Tesla T4 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, primals_7, primals_1, primals_3, primals_5, primals_8, convolution, cat, tangents_1):
        sum_1 = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(tangents_1, cat, primals_5, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  tangents_1 = cat = primals_5 = None
        getitem = convolution_backward[0]
        getitem_1 = convolution_backward[1];  convolution_backward = None
        slice_2 = torch.ops.aten.slice.Tensor(getitem, 1, 32, 64);  getitem = None
        sum_2 = torch.ops.aten.sum.dim_IntList(slice_2, [0, 2, 3])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(slice_2, convolution, primals_3, [32], [4, 4], [0, 0], [1, 1], True, [0, 0], 1, [True, True, False]);  slice_2 = convolution = primals_3 = None
        getitem_3 = convolution_backward_1[0]
        getitem_4 = convolution_backward_1[1];  convolution_backward_1 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(getitem_3, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(getitem_3, primals_8, primals_1, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  getitem_3 = primals_8 = primals_1 = None
        getitem_7 = convolution_backward_2[1];  convolution_backward_2 = None
        return [getitem_7, sum_3, getitem_4, sum_2, getitem_1, sum_1, None, None]
        
def load_args(reader):
    reader.symint(99)  # primals_7
    buf0 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf0, (128, 32, 4, 4), requires_grad=True, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1, (128, 32, 4, 4), requires_grad=True, is_leaf=True)  # primals_3
    buf2 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf2, (32, 64, 1, 1), (64, 1, 64, 64), requires_grad=True, is_leaf=True)  # primals_5
    buf3 = reader.storage(None, 830472192, device=device(type='cuda', index=0))
    reader.tensor(buf3, (99, 32, 256, 256), is_leaf=True)  # primals_8
    buf4 = reader.storage(None, 207618048, device=device(type='cuda', index=0))
    reader.tensor(buf4, (99, 128, 64, 64), is_leaf=True)  # convolution
    buf5 = reader.storage(None, 1660944384, device=device(type='cuda', index=0))
    reader.tensor(buf5, (99, 64, 256, 256), is_leaf=True)  # cat
    buf6 = reader.storage(None, 830472192, device=device(type='cuda', index=0))
    reader.tensor(buf6, (99, 32, 256, 256), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    run_repro(mod, load_args, accuracy=False, command='minify', save_dir='/home/vboza/dev-ai/torch_compile_debug/run_2023_11_29_12_29_17_578281-pid_3430/minifier/checkpoints', tracing_mode='real', check_str=None)