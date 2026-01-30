import torch


def fn_torch(x):
    return x * x


fn_compiled = torch.compile(fn_torch)


x = torch.tensor([2.0, 3.0, 4.0])
result = fn_compiled(x)
print(result)