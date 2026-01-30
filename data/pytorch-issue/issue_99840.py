import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.y = 0.1

    def forward(self, x):
        out = self.conv_transpose(x)
        out = torch.gt(out, 0)
        out = torch.mul(out, self.y)
        return out

x = torch.randn(1, 3, 4, 4)

func = Model()

res1 = func(x)
print(torch.isnan(res1).any())
# tensor(False)

jit_func = torch.compile(func)
res2 = jit_func(x)
print(torch.isnan(res2).any())
# tensor(True)