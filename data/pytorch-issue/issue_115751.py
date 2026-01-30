import torch

def func(x, a):
    n = (a * 1.234) // 8.234
    y = x + n
    return y

cfunc = torch.compile(func, dynamic=True, fullgraph=True)

device = "cuda"
x = torch.tensor(0, dtype=torch.float32, device=device)
a = 33

out = cfunc(x, a)
expected = func(x, a)
torch.testing.assert_close(out, expected)