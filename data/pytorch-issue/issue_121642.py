import torch
import torch.nn as nn

class Bar(torch.nn.Module):
    def forward(self, x, y):
        return x + y[1:]

dx = Dim("dx", min=1, max=3)
ep = export(
    Bar(),
    (torch.randn(2, 2), torch.randn(3, 2)),
    dynamic_shapes=({0: dx, 1: None}, {0: dx+1, 1: None})
)
print(ep.range_constraints)

{s0: ValueRanges(lower=2, upper=3, is_bool=False), s0 + 1: ValueRanges(lower=3, upper=4, is_bool=False)}

{s0: ValueRanges(lower=1, upper=3, is_bool=False), s0 + 1: ValueRanges(lower=2, upper=4, is_bool=False)}