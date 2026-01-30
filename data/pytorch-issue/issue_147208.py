import torch

def f(*args):
    sym_5, sym_6, sym_7 = args

    var_279 = torch.quantize_per_tensor(torch.randn((100,)), scale=sym_5, zero_point=sym_6, dtype=sym_7)
    var_374 = torch.flip(var_279, dims=(0,))
    return var_374

res = f(3., 10, torch.quint2x4)
print(res)