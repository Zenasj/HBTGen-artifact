import torch

t = torch.tensor([[1,2],[3,4]])
index = torch.tensor([[0]])
torch.gather(t, 1, index)

torch.gather(torch.randn(3,3), 0, torch.tensor([0]))
tensor([-0.1960])