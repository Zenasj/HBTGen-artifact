import torch.nn as nn

py
import torch

torch.manual_seed(420)

class M(torch.nn.Module):

    def __init__(self):
        super(M, self).__init__()
        self.convt = torch.nn.ConvTranspose2d(4, 4, 3, stride=1, padding=1)

    def forward(self, in_vals):
        v1 = self.convt(in_vals) + 3
        v2 = v1.clamp(0, 6)
        v3 = v2 / 6
        return v3

func = M().to('cpu').eval()

x = torch.randn(1, 4, 1, 1)

with torch.no_grad():
    res1 = func(x) # without jit
    print(res1)

    jit_func = torch.compile(func)
    res2 = jit_func(x)
    print(res2)
    # RuntimeError: y.get_desc().is_nhwc() INTERNAL ASSERT FAILED at "../aten/src/ATen/native/mkldnn/Conv.cpp":741, please report a bug to PyTorch.