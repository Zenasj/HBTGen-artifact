import torch
a = torch.tensor(1., requires_grad=True)
(a * torch.eye(3)).eig(eigenvectors=True)[1].sum().backward()