import torch
x_cpu = torch.tensor([], dtype=torch.bool)
x_mps = x_cpu.to("mps")
assert x_cpu.all() == x_mps.all().cpu()