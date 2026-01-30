import torch

a = torch.randint(0, 2, size=(65537, 50)).cuda()
a.mode(-1)

b = torch.randint(0, 2, size=(65532, 50)).cuda()
b.mode(-1) # --> fast, even with context initialization time
a = torch.randint(0, 2, size=(65537, 50)).cuda()
a.mode(-1) # --> super slow