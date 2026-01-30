import torch

t = torch.tensor([1, 2, 3, 4], device='mps')
t[2:] <= t[:2] # Expected [False, False] but got [True, True].