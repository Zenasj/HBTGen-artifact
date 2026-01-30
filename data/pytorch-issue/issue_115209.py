import math
import torch

def func(x):
    return (
        torch.tensor([math.ceil(x.item())]),
        torch.tensor([math.floor(x.item())]),
    )

x = torch.randn(1, dtype=torch.float32)
ep = torch.export.export(func, args=(x,))
ep_decomp = ep.run_decompositions()