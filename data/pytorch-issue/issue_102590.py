import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = torch.split(x, 2, 1)
        v2 = v1[0]
        v3 = v1[1]
        v4 = torch.cat([v2, v3, v2, v3], 1)
        return v4

func = Model().to('cuda')
x = torch.randn(1, 3, 64, 64).cuda()

with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)

    res1 = func(x) # without jit
    print(res1)

    res2 = jit_func(x)
    # AssertionError: TritonKernel.indexing assumes YBLOCK divides config.triton.max_block["Y"] but YBLOCK=2048 and config.triton.max_block["Y"]=1024 (cfg={'XBLOCK': 2, 'YBLOCK': 2048}).