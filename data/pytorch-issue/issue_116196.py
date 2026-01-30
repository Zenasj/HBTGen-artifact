import torch
aten = torch.ops.aten

def func(x, a):
    return aten.div(x * 0.5, a, rounding_mode=None)


cfunc = torch.compile(func, dynamic=True, fullgraph=True)
device = "cpu"
x = 124
a = 33
out = cfunc(x, a)
expected = func(x, a)
torch.testing.assert_close(out, expected)