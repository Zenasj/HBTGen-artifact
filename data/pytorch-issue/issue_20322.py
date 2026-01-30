import torch
a = torch.randn(3, 5)
c = torch.zeros(3)
a.index_copy_(dim=1, index=torch.tensor([3]), source=c)

a.index_copy_(dim=1, index=torch.tensor([3]), source=c.unsqueeze(1))