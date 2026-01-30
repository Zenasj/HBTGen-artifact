import torch

@torch.compile
def f(input):
    var_17 = torch.slice_copy(input, dim=0, start=449, end=None, step=9223372036854775807)
    return torch.reciprocal(var_17)

input = torch.randn((875,))
res = f(input)
print(res)