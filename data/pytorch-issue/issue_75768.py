import torch.nn as nn

import torch

def fn(x: int):
    y = torch.zeros((x, x+2)).cuda()
    for i in range(2):
        inp = torch.rand((x, x+i)).cuda()
        weight = torch.rand((x+2, x+i)).cuda()
        bias = torch.rand((x, x+2)).cuda()
        y += torch.sin(torch.nn.functional.linear(inp, weight, bias))
    return y

fn.__disable_jit_function_caching__ = True

with torch.jit.fuser("fuser2"):
    fn_s = torch.jit.script(fn)
    fn_s(5)
    fn_s(5)