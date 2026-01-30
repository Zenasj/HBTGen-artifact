import torch

x = torch.randn(4, 8)  # [s0, s1]
y = torch.randn(32)  # [s2]
out = x.reshape(-1) + y
# this emits Eq(s0 * s1, s2), and we represent y's shape as [s0*s1] in the graph.