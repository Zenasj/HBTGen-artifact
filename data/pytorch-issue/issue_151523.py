import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config

config.fallback_random = True
torch.set_grad_enabled(False)
import os
os.environ['TORCHDYNAMO_VERBOSE'] = '1'


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(1, 7), stride=(2, 1), padding=0)

    def forward(self, x, weight):
        x = self.conv(x)
        x = F.hardshrink(x, lambd=0)
        x = x.view(x.size(0), -1)
        x = torch.mv(weight, x[0])
        return x


model = Model()


x = torch.randn(2, 3, 127, 255)
weight = torch.randn(10, 254976)

inputs = [x, weight]


def run_test(model, inputs, backend):
    torch.manual_seed(0)
    if backend != "eager":
        model = torch.compile(model, backend=backend)
    try:
        output = model(*inputs)
        print(f"succeed on {backend}")
    except Exception as e:
        print(e)


run_test(model, inputs, 'eager')
run_test(model, inputs, 'inductor')