import torch

torch._refs.tensor([])  # error
torch.tensor([])  # OK