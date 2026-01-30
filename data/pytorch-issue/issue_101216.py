import torch.nn as nn
import math

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 * 0.7071067811865476
        v3 = torch.sign(v2)
        v4 = v3 * v3
        v5 = torch.tanh(v4)
        return v5

func = Model().to('cuda')

x = torch.randn(1, 3, 64, 64).cuda()

with torch.no_grad():
    func = func.eval()
    res1 = func(x) # without jit
    print(res1)

    jit_func = torch.compile(func)
    res2 = jit_func(x)
    # torch._dynamo.exc.BackendCompilerFailed: backend='debug_wrapper' raised:
    # CompilationError: at 19:25:    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    #     tmp1 = tl.load(in_ptr0 + (x1), None)
    #     tmp2 = tmp0 + tmp1
    #     tmp3 = 0.7071067811865476
    #     tmp4 = tmp2 * tmp3
    #     tmp5 = 0 < tmp4
    #     tmp6 = tl.where(tmp5, 1, 0)
    #     tmp7 = tmp4 < 0
    #     tmp8 = tl.where(tmp7, 1, 0)
    #     tmp9 = tmp6 - tmp8
    #     tmp10 = tmp9 * tmp9
    #     tmp11 = tl.math.tanh(tmp10)
    #                          ^
    # ValueError('input arg type does not match.Expect one of dict_keys([(triton.language.fp32,), (triton.language.fp64,)]), got (triton.language.int32,)')