import torch.nn as nn

import torch

torch.manual_seed(0)
torch.set_grad_enabled(False)
from torch._inductor import config

config.fallback_random = True


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.sum(dim=-1)
        x = self.softmax(x)
        return x


model = Model()

x = torch.randn(8, 8, 2)

inputs = [x]

try:
    output = model(*inputs)
    print("succeed on eager")
except Exception as e:
    print(e)

try:
    c_model = torch.compile(model)
    c_output = c_model(*inputs)
    print("succeed on inductor")

except Exception as e:
    print(e)