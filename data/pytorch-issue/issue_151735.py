import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config

config.fallback_random = True
torch.set_grad_enabled(False)
torch.manual_seed(0)


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x_flat = x.flatten()[0:5]
        y = torch.ones_like(x_flat)
        x = torch.vdot(x_flat, y)
        return x


model = Model()


x = torch.tensor([[0.0001, 1000000.0], [-1000000.0, 0.1]])  # 0.1 is the key factor

inputs = [x]


def run_test(model, inputs, device, backend):
    torch.manual_seed(0)
    model = model.to(device)
    inputs = [x.to(device) for x in inputs]
    if backend != "eager":
        model = torch.compile(model, backend=backend)
    torch.manual_seed(0)
    output = model(*inputs)
    return output


device = 'cuda'
output = run_test(model, inputs, device, 'eager')
c_output = run_test(model, inputs, device, 'inductor')
print(output)
print(c_output)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config

config.fallback_random = True
torch.set_grad_enabled(False)
torch.manual_seed(0)


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x_flat = x.flatten()[0:5]
        y = torch.ones_like(x_flat)
        x = torch.vdot(x_flat, y)
        return x


model = Model()


x = torch.tensor([[0.0001, 1000000.0], [-1000000.0, 0.1]])  # 0.1 is the key factor

inputs = [x]


def run_test(model, inputs, device, backend):
    torch.manual_seed(0)
    model = model.to(device)
    inputs = [x.to(device) for x in inputs]
    if backend != "eager":
        model = torch.compile(model, backend=backend)
    torch.manual_seed(0)
    output = model(*inputs)
    return output


device = 'cuda'
output = run_test(model, inputs, device, 'eager')
c_output = run_test(model, inputs, device, 'aot_eager_decomp_partition')
fp64 = run_test(model.to(dtype=torch.float64), [x.to(dtype=torch.float64)], device, 'eager')
print(output)
print(c_output)
print(fp64)

print(torch._dynamo.utils.same(output, c_output, fp64))