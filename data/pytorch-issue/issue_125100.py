import torch

device = torch.device("mps")

x = torch.tensor([[1, 0, 2, 0, -100, -100, -100],
                  [1, 0, 2, 0, -100, -100, -100]])
mask = torch.tensor([[True, True, True, True, False, False, False],
                     [True, True, True, True, False, False, False]])

x = x.to(device)
mask = mask.to(device)

print(x[mask])