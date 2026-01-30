import torch

x = torch.zeros([1, 2])
y = torch.zeros([1, 3])
torch.stack([x, y]) # Errors due to size differences