import torch
d = torch.distributions.Dirichlet(torch.sigmoid(torch.randn(3, 4, requires_grad=True))).rsample()
torch.mean(d).backward() # works!
d = torch.distributions.Dirichlet(torch.sigmoid(torch.randn(3, 4, requires_grad=True).cuda())).rsample()
torch.mean(d).backward() # throws above exception