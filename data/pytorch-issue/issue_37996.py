import torch
torch.scatter(torch.ones(4), 0, torch.tensor([0, 1], dtype=torch.int64), torch.zeros(4))