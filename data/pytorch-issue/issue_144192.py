import torch
a = torch.rand(2, 2, device="cuda")
b = torch.tensor(1e-3)

torch.add(a, b)
torch.addcdiv(a, a, b)  # used to fail, now works