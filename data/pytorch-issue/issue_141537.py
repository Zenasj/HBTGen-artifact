import torch

torch.save(1., "ff")
torch.load("ff", weights_only=True)

torch.save(1.j, "ff")
torch.load("ff", weights_only=True)