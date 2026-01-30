import torch
t=torch.empty(10, dtype=torch.float8_e5m2)
torch.save(t, "/tmp/t")