import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, (2, 3), stride=(1, 2), padding=0)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + torch.ones(v1.shape, dtype=torch.double)
        return v2

func = Model().to('cpu')

x = torch.randn(1, 3, 64, 64)

with torch.no_grad():
    func = func.eval()
    res1 = func(x) # without jit
    print(res1)

    jit_func = torch.compile(func)
    res2 = jit_func(x)
    # torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
    # AssertionError: Pointwise(
    #   'cpu',
    #   torch.float64,
    #   def inner_fn(index):
    #       i0 = index
    #       tmp0 = ops.load(arg1_1, i0)
    #       tmp1 = ops.to_dtype(tmp0, torch.float64)
    #       return tmp1
    #   ,
    #   ranges=[8],
    #   origin_node=None,
    #   origins={full, fn}
    # )