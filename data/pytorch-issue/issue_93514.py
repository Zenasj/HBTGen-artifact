import torch

inp = torch.rand(3, 10)

def f(inp):
    return torch.cat(
        [torch.relu(x) for x in inp], dim=0
    )

opt_f = torch.compile(f)
print(opt_f(inp))
print(opt_f(torch.rand(5, 20)))