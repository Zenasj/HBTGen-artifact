import torch
data = torch.zeros([2, 4], dtype=torch.int64).to('mps')
out = torch.abs(data)