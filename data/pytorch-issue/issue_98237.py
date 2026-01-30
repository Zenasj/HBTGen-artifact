import torch
from torch._inductor.compile_fx import compile_fx_inner
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

def f(x):
    isinf = torch.isinf(x)
    return [isinf]

with FakeTensorMode():
    inp = torch.ones(2048, device='cuda')
    fx_g = make_fx(f)(inp)
    fx_inner = compile_fx_inner(fx_g, [inp])