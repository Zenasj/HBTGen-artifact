import torch

def f(*args):
    sym_0, sym_1, sym_2, sym_3 = args

    var_228 = torch.arange(start=sym_0, end=sym_1, dtype=sym_2)
    return torch.sum(var_228, dim=sym_3)

res = f(300, 1024, torch.float16, (0,))
print(res)

res = torch.compile(f)(300, 1024, torch.float16, (0,))
print(res)