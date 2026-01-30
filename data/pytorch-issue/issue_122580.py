import torch
def do_ds(x, y):
    return torch.diagonal_scatter(x, y)

x=torch.ones(10, 10, dtype=torch.int64)
y=torch.tensor([ 1,  2, -8,  8,  5,  5, -7, -8,  7,  0])
dsc = torch.compile(do_ds)
assert torch.allclose(torch.diagonal_scatter(x, y), dsc(x, y))