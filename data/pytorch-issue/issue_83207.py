import torchvision
import numpy as np

import torch
from torchvision.models import resnet18

from functorch._src.compilers import nop
from functorch._src.aot_autograd import aot_module
from functorch.compile import config

config.use_functionalize = True

model = resnet18().cuda().half().to(memory_format=torch.channels_last)
input = torch.randn(256, 3, 224, 224, device='cuda', dtype=torch.float16) \
             .to(memory_format=torch.channels_last).detach().requires_grad_(True)
input_expected = input.clone().detach().requires_grad_(True)

fn = aot_module(model, nop)
out = fn(input)
out_expected = model(input_expected)
print(torch.allclose(out, out_expected))

out.sum().backward()
out_expected.sum().backward()
print(torch.allclose(input.grad, input_expected.grad))

@register_decomposition(aten.new_zeros, aot_autograd_decompositions)
def new_zeros(inp, size, dtype=None, layout=None, device=None, pin_memory=None):
    return torch.zeros(size, dtype=inp.dtype, device=inp.device)