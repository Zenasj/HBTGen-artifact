import torch
ints = torch.arange(24, device='cuda:0', dtype=torch.float32).reshape(8,3)
inds = torch.tensor([1, 1, 0, 2, 2, 2, 0, 3], device='cuda:0')
inds = inds.unsqueeze(1).expand(-1, 3)
torch.scatter_reduce(ints, 0, inds, reduce='mean')