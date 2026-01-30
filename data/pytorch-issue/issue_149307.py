import torch

print(torch.__version__)

def f(*args):
    sym_0, sym_1, sym_2, sym_3 = args

    var_941 = torch.arange(start=sym_0, end=sym_1, step=1)
    return torch.isin(var_941, sym_2, assume_unique=sym_3)

res = f(0, 1024, 1, True,)
print('eager: ', res)

res = torch.compile(f)(0, 1024, 1, True,)
print('inductor: ', res)