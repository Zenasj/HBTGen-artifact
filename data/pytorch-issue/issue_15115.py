import torch
mu = torch.ones(1, requires_grad=True)
x = torch.randn(1)
loss = 0
for _ in range(3):
    x.detach_()
    new_calc = torch.exp(mu)
    x.copy_(new_calc)
    loss += (x * 2).sum() #broken
    # loss += torch.mul(x, 2) # broken
    # loss += (x * torch.tensor([2.])).sum() #broken
    # loss += ((x+0) * torch.tensor([2.])).sum() #works
    # loss += (x+0)*torch.randn(1) #works
    # loss += x+torch.randn(1) #works
    # loss += torch.exp(x) #works
loss.backward()

CONTRIBUTING.MD