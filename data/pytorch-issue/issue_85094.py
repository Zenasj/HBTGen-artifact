import torch

t = torch.randn((3,))
torch.tensor(t).unsqueeze_(1)

t = torch.randn((3,))
t.clone().detach().unsqueeze_(1)