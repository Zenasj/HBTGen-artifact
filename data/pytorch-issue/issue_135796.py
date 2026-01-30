import torch.nn as nn

import torch
from triton.testing import do_bench

torch.set_default_device("cuda")

def f(x, y):
    return x.repeat([2])

x = torch.randn(20)
torch._dynamo.mark_dynamic(x, 0)
so_path = torch._export.aot_compile(f, (x, True))
fn = torch._export.aot_load(so_path, "cuda")
out = fn(torch.randn(40), True) 
assert out.shape[0] == 80 # Returns 40(!) instead of 80

class MyModule(torch.nn.Module):
    def forward(self, x):
        return x.repeat([2])

x = torch.randn(20)
torch._dynamo.mark_dynamic(x, 0)
ep = torch.export.export(MyModule(), (x,))
so_path =  torch._inductor.aoti_compile_and_package(ep)
fn = torch._inductor.aoti_load_package(so_path)
out = fn(torch.randn(40), True) 
assert out.shape[0] == 80 # Returns 40(!) instead of 80