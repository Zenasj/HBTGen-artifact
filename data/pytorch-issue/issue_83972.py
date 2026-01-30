import torch

x = torch.Tensor([[1.,2.]])
y = torch.Tensor([0.,0.])
z = torch.squeeze(x, out=y)

x = torch.Tensor([[1.,2.]])
z = torch.squeeze(x)