import torch

switch   = torch.tensor([True, False])
vals     = torch.tensor([1., 2.], dtype=torch.float32)
selected = torch.where(switch, vals, 0.)