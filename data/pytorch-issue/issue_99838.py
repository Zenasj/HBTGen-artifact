import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out_conv = self.conv(x)
        out = torch.add(out_conv, 1.0)
        return out

x = torch.rand([1, 3, 224, 224])

func = Model().to('cpu')

res1 = func(x)
print(res1)

with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)
    res2 = jit_func(x)
    print(res2)
    # AttributeError: 'float' object has no attribute 'meta'