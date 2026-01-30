import torch

mu = torch.randn(8)
sigma = torch.eye(8).expand(6, 1, 8, 8).contiguous().requires_grad_()
x_repeat = torch.randn(6,8000,8)
pdf = MultivariateNormal(mu, sigma).log_prob(x_repeat).t()
pdf.sum().backward()

sigma = torch.eye(8).expand(6, 1, 8, 8).contiguous().requires_grad_()
x_repeat = torch.randn(8000, 6, 1, 8)