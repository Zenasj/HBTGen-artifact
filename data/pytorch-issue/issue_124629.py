import torch

def fn(x):
    return torch.cat(torch.split(x, 4, 1), torch.tensor(1))

x = torch.randn(2, 32, 32, 16)
fn = torch.compile(fn, fullgraph=True)
out = fn(x)
print(out.shape)