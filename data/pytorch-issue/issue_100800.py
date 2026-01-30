import torch

input_tensor = torch.ones(3, 3)

def f(x):
    return torch.where(torch.ones_like(x).to(torch.bool), torch.zeros_like(x), torch.ones_like(x)* 2)

res1 = f(input_tensor)
print(res1)

jit_func = torch.compile(f)
res2 = jit_func(input_tensor)
print(res2)