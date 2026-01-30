import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1)

    def forward(self, x):
        h = self.conv(x)
        h = torch.mul(h, 3)
        a = torch.clamp_min(torch.clamp_max(h, 6.0), 0.0)
        b = torch.add(a, 3)
        v1 = h / 6.0
        v2 = torch.div(42, v1)
        return v1 + v2

x = torch.randn(1, 3, 224, 224).cuda()
func = Model().to('cuda')

jit_func = torch.compile(func)

res1 = func(x) # without jit
print(res1)

res2 = jit_func(x)
# torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
# CompilationError: at 16:19:    xmask = xindex < xnumel
#     x0 = xindex
#     tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
#     tmp1 = tl.load(in_ptr0 + (0))
#     tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
#     tmp3 = tmp0 + tmp2
#     tmp4 = 3.0
#     tmp5 = tmp3 * tmp4
#     tmp6 = 6.0
#     tmp7 = tmp5 / tmp6
#     tmp8 = 42.0
#     tmp9 = tmp8 // tmp7
#                    ^
# AssertionError()