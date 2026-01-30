import torch
device = torch.device("cpu")
# device = torch.device("cuda")

x = torch.arange(10, device=device)
ind = torch.arange(3, 8, device=device)
tmp = torch.zeros(5, 2, device=device, dtype=ind.dtype)
out = tmp[:, 0]

torch.searchsorted(x, ind, out=out)
print(out)
print(tmp)