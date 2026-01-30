import torch.nn as nn
import torchvision

from typing import List
import torch
from torch._subclasses import FakeTensorMode, FakeTensor
from functorch.compile import aot_function, print_compile, config, aot_module
from functorch import make_functional_with_buffers, vmap, combine_state_for_ensemble
from functorch._src.named_members_polyfill import _named_parameters, _named_buffers
from torchvision.models import resnet18

g = {}
def fake_wrapper(gtype):
    def fake_compiler(fx_g, inps):
        print(fx_g.code)
        nonlocal gtype
        g[gtype] = fx_g
        return fx_g
    return fake_compiler


inp = torch.randn(32, 3, 224, 224, dtype=torch.float32).cuda()
targets = torch.zeros(32, dtype=int).cuda()

b_models:List[torch.nn.Module] = [resnet18().cuda() for _ in range(5)]

func_model, params, buffers = combine_state_for_ensemble(b_models)
for p in params:
    p.requires_grad = True

def compute_loss(weights, buffers, batch, targets):
    output = func_model(weights, buffers, batch)
    loss = torch.nn.functional.nll_loss(output,targets)
    return loss
parallel_func = vmap(compute_loss, in_dims=(0,0,None, None))
aot_func = aot_function(parallel_func, fake_wrapper("forward"), fake_wrapper("backward"))
out = aot_func(params, buffers, inp, targets)
out.mean().backward()