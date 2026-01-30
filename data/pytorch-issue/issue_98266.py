import torch

x = torch.range(1, 3)[None, :]
y = x.to("mps")

x.flip(0)  # tensor([[1., 2., 3.]])
y.flip(0)  # tensor([[3., 2., 1.]], device='mps:0')