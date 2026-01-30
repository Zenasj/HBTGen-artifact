import torch
x = torch.rand(32, device="mps")
y = torch.rand(32, device="mps")
x[3] = torch.nan
y[5] = torch.nan
print(x.isnan().any().item(), torch.minimum(x, y).isnan().any().item())