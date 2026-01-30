import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self, min_value, max_value):
        super().__init__()
        self.convT = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)
        self.min_val = min_value
        self.max_val = max_value

    def forward(self, x):
        v1 = self.convT(x)
        v2 = torch.clamp(v1, self.min_val, float('inf'))
        v3 = torch.clamp(v2, float('-inf'), self.max_val)
        return v3

func = Model(-0.5, 0.5).to('cpu')

x = torch.randn(1, 3, 64, 64)

with torch.no_grad():
    func = func.eval()
    res1 = func(x) # without jit
    print(res1)

    jit_func = torch.compile(func)
    res2 = jit_func(x)
    # buf1 = torch.ops.mkldnn._convolution_transpose_pointwise(buf0, arg0_1, arg1_1, (1, 1), (0, 0), (1, 1), (1, 1), 1, 'hardtanh', [-0.5, inf], '')
    # NameError: name 'inf' is not defined