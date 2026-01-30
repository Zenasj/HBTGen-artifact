# nvfuser.py
import torch

a = torch.rand((2, 2)).cuda()
b = torch.rand((2, 2)).cuda()

def fn(x, y):
    return x.sin() + y.exp()

fn_s = torch.jit.script(fn)
fn_s(a, b)
fn_s(a, b)