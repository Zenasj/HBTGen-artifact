import torch

device = torch.device("mps:0")
x = torch.tensor([[-5.], [0.], [5.]], device=device)
inds = torch.tensor([[0], [1], [2]], device=device)
torch.take_along_dim(x, inds, 0)