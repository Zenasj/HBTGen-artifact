import torch

def fn(x):
    a = torch.sin(x)
    b = torch.cos(a)
    return b

inp = torch.randn(10, 100)
fn_tvm = torch.compile(fn, backend="tvm")

print(torch.allclose(fn(inp), fn_tvm(inp)))