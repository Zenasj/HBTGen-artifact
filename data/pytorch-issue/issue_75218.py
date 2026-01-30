import torch
from torch.utils._pytree import tree_map

print(tree_map(lambda a: None, torch.cummin(torch.randn(3), 0)))