import torch.nn as nn
import numpy as np
import random

py
import torch

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        exponent = 0
        self.exponent = exponent

    def forward(self, input):
        exponent = self.exponent
        fn_res = torch.pow(input, exponent, )
        fn_res = torch.sin(fn_res)
        return fn_res

fn = M().to('cpu')

torch.random.manual_seed(15316)
inp = torch.empty([5], dtype=torch.bfloat16)
inp.uniform_(-64, 127)

print('normal')
print(fn(inp))

jit_fn = torch.jit.trace(fn, inp)
print('jit')
print(jit_fn(inp))