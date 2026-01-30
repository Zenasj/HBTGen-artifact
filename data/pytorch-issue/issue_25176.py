import torch
import numpy as np

t = torch.tensor(range(5))
mask = np.array([False, True, True, False, True])

print(t[mask.astype(np.bool)])
print(t[mask.astype(np.uint8)])

t = torch.tensor(range(5))
mask = torch.tensor([False, True, True, False, True])

print(t[mask.to(torch.bool)])
print(t[mask.to(torch.uint8)])