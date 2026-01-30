import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config
import os
config.fallback_random = True
torch.set_grad_enabled(False)
torch.manual_seed(0)



class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.shrink = nn.Tanhshrink()

    def forward(self, x):
        x = self.shrink(x)
        x = torch.atan2(x, x)
        return x


model = Model()


x = torch.randn(1, 3, 64, 64)

inputs = [x]



def run_test(model, inputs, backend):
    if backend != "eager":
        model = torch.compile(model, backend=backend)
    torch.manual_seed(0)
    output = model(*inputs)
    return output


output = run_test(model, inputs, 'eager')
c_output = run_test(model, inputs, 'inductor')

print(torch.allclose(output, c_output, 1e-3, 1e-3, equal_nan=True))
print(torch.max(torch.abs(output - c_output)))