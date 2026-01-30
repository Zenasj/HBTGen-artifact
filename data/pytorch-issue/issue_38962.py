import torch

python
@torch.jit.script
def f(x: torch.Tensor):
    mask = torch.ones(x.shape, dtype=torch.bool)
    x[mask] = 2

python
x[mask] = torch.tensor(2)
# or
x.masked_fill_(mask, 2)