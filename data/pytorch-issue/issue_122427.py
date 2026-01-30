import torch

p = torch.polar(torch.ones_like(torch.empty(2, 3)), torch.empty(2, 3))
p.index_select(0, torch.arange(0, 2, dtype=torch.int64).to("mps"))