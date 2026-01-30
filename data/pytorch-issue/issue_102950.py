import torch

def fn(x):
    return torch.cat(torch.split(x, 1), dim=-1)


x = torch.rand(2, 3).to("cuda")
print(fn(x))
opt_fn = torch.compile(fn)
print(opt_fn(x))