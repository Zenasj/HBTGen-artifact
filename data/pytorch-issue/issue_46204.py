import torch

a = torch.zeros(2, requires_grad=True)
b = a[1]
b.add_(1) # wouldn't fail unless you called b.backward() before, now this line itself fails