import torch.nn as nn

py
import torch

torch.manual_seed(420)

width = 224
height = 224
channels = 3
batch_size = 1
x = torch.randn(batch_size, channels, height, width)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv(x)
        out = torch.add(out, out)
        return out

func = Model()

with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)

    res1 = func(x) # without jit
    print(res1)

    res2 = jit_func(x)
    # RuntimeError: Tried to erase Node _convolution_pointwise but it still had 1 users in the graph: {fn: None}!