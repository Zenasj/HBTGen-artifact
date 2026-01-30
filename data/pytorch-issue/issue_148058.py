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
        x = F.gumbel_softmax(x, tau=1.0, hard=True)
        x = torch.where(x > 0.5, x, torch.zeros_like(x))
        x = torch.scatter(x, dim=1, index=torch.ones(1, 2, dtype=torch.long), src=torch.ones_like(x))
        return x


model = Model()


x = torch.randn(1, 2)

inputs = [x]


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