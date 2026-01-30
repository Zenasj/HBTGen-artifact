import torch

device = torch.device("cuda:1")
dtype = torch.float32

mean = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
sd = torch.diag_embed(torch.tensor([1.0, 1.0], dtype=dtype, device=device))
distribution = torch.distributions.MultivariateNormal(mean, sd)

data1 = torch.randn([524280,2], dtype=dtype, device=device)
logprob = distribution.log_prob(data1)