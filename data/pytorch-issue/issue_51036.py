import torch.nn as nn

python
import torch
import torch.nn.functional as F


for seed in range(0, 20):
    torch.manual_seed(seed)

    in_shape = torch.randint(low=10, high=50, size=(2,)).tolist()
    in_map = torch.randn(2, 256, *in_shape, device='cuda', requires_grad=True)

    out_shape = torch.randint(low=10, high=50, size=(2,)).tolist()
    inter_map = F.interpolate(in_map, size=tuple(out_shape), mode='nearest')

    output_map = torch.mul(inter_map, 0.0)
    output_map.sum().backward()

    max_grad = in_map.grad.abs().max().item()
    print(f"({seed:02d}) Max grad: {max_grad}")

import torch
import torch.nn.functional as F


in_shape = (45,18) #torch.randint(low=10, high=50, size=(2,)).tolist()
in_map = torch.ones(2, 256, *in_shape, device='cuda', requires_grad=True)

out_shape = torch.randint(low=10, high=50, size=(2,)).tolist()
out_shape = (14,24)
inter_map = F.interpolate(in_map, size=tuple(out_shape), mode='nearest')
output_map = torch.mul(inter_map, 0.0)
output_map.sum().backward()

max_grad = in_map.grad.abs().max().item()
print(f"Max grad: {max_grad}")