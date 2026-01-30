import torch
a=torch.ones([838860800], dtype = torch.float, device="cuda")
print(a.mean())