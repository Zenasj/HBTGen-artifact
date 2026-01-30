import torch

from torch.func import vmap
from C_extension import constrain_make_charts

cond_array = torch.tensor(
    [
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ],
    dtype=torch.int64,
).reshape(-1, 4)

cond_array = (cond_array * torch.tensor([1, 2, 4, 8])).sum(dim=1, keepdim=True)

func = vmap(constrain_make_charts, in_dims=0)

values = constrain_make_charts(cond_array)
print(values)
func(cond_array)