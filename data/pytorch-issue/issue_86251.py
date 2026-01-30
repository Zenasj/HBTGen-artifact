import torch.nn as nn
import numpy as np
import random

py
import torch

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = -1
        self.keepdim = False

    def forward(self, input):
        dim = self.dim
        keepdim = self.keepdim
        input = torch.sub(input, torch.tensor(9, dtype=torch.float32, device='cuda'))
        fn_res = torch.mean(input, dim, keepdim=keepdim, )
        fn_res = torch.sub(fn_res, torch.tensor(-3, dtype=torch.float32, device='cuda'))
        return fn_res

fn = M().to('cuda')
torch.random.manual_seed(56288)
inp = torch.empty([1, 64], dtype=torch.float32, memory_format=torch.contiguous_format)
inp.uniform_(-128, 63)

def clone(tensor):
    return tensor.clone().to('cuda').requires_grad_()

jit_fn = torch.jit.trace(fn, clone(inp))

from torch.autograd.functional import jacobian
jit_fn(clone(inp))

jacobian(jit_fn, (clone(inp), ))

jacobian(jit_fn, (clone(inp), ), vectorize=True, strategy='forward-mode')