import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config

config.fallback_random = True
torch.set_grad_enabled(False)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = F.fractional_max_pool2d(x, kernel_size=3, output_size=(1, 1))
        return x


model = Model()


x = torch.randn(1, 1, 6, 6).cuda()


inputs = [x]


def run_test(model, inputs, backend):
    torch.manual_seed(0)
    if backend != "eager":
        model = torch.compile(model, backend=backend)
    try:
        output = model(*inputs)
        print(output)
        print(f"succeed on {backend}")
    except Exception as e:
        print(e)


run_test(model, inputs, 'eager')
run_test(model, inputs, 'inductor')