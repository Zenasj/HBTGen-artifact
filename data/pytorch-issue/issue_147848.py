import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config

config.fallback_random = True
torch.set_grad_enabled(False)


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fold = torch.nn.Fold(output_size=(4, 4), kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        x = self.fold(x)
        return x


model = Model()


x = torch.randn(1, 4, 4)

inputs = [x]


def run_test(model, inputs, backend):
    torch.manual_seed(0)
    if backend != "eager":
        model = torch.compile(model, backend=backend)
    try:
        c_output = model(*inputs)
        print(c_output)
    except Exception as e:
        print(e)


run_test(model, inputs, 'eager')
run_test(model, inputs, 'inductor')