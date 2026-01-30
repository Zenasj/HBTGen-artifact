import torch.nn as nn
import numpy as np
import random

py
import torch

torch.random.manual_seed(3564)
def get_fn():
    arg_class = torch.nn.CTCLoss()
    inp1 = torch.randint(-16, 1, [16, 30], dtype=torch.int64, device='cuda')
    inp2 = torch.randint(-4, 2, [16], dtype=torch.int64, device='cuda')
    inp3 = torch.randint(-1, 1, [16], dtype=torch.int64, device='cuda')
    def fn(inp):
        fn_res = arg_class(inp, inp1, inp2, inp3)
        return fn_res
    return fn

fn = get_fn()
inp = torch.empty([1, 16, 20], dtype=torch.float32)
inp.uniform_(-32, 15)
inp_a = inp.clone().to('cuda')
inp_b = inp.clone().to('cuda')
inp_c = inp.clone().to('cuda')

res = fn(inp_a)

jit_fn = torch.jit.trace(fn, (inp_b, ))
jit_res = jit_fn(inp_c)

print(res) # tensor(12.9426, device='cuda:0')
print(jit_res) # tensor(14.2359, device='cuda:0')