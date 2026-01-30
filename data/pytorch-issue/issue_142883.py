import torch
from torch._higher_order_ops.associative_scan import associative_scan

x = torch.randn(10, device='cuda')
y1 = associative_scan(lambda acc, curr: acc + torch.abs(curr), x, dim=-1, combine_mode="pointwise")