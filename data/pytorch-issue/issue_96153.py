import torch

t = torch.ones((2,6,), device='mps')[1].reshape(2,3)
print("No error yet")
t = t + 1

import torch

t = torch.ones((2,6,), device='mps')[1].clone().reshape(2,3)
print("No error yet")
t = t + 1