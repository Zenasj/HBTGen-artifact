import torch

def fn(x):
    a = torch.sin(x)
    b = torch.cos(a)
    return b

inp = torch.randn(10, 100)
fn_tvm = torch.compile(fn, backend="tvm", options={"scheduler": "meta_schedule", "trials": 100})

print(torch.allclose(fn(inp), fn_tvm(inp)))