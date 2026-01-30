import torch


def func(x, a):
    n = (a * 1.0) // 8.0
    y = x + n
    return y

cfunc = torch.compile(func, dynamic=True, fullgraph=True)

device = "cpu"
x = torch.tensor(0, dtype=torch.float32, device=device)
a = 12

out = cfunc(x, a)
expected = func(x, a)
torch.testing.assert_close(out, expected)