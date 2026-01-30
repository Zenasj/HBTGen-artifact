import torch
from torch.distributions.constraints import lower_cholesky

x = torch.tensor([[1.4142]])
print("CPU:", lower_cholesky.check(x).item())
print("MPS:", lower_cholesky.check(x.to("mps")).item())