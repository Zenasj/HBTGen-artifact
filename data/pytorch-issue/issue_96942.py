import torch

device = torch.device("mps")
x = torch.rand(1, 4, 4).to(device)
shifted_x = torch.roll(x, shifts=(-1, -1), dims=(1, 2))

print(shifted_x)