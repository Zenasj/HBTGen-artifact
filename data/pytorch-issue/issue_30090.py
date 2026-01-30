import numpy as np

import torch
import torch.distributions as dist

# built in kl divergence
a = dist.Dirichlet(torch.ones(5)).sample()
a.requires_grad = True
d1 = dist.Categorical(a)
d2 = dist.Categorical(torch.ones(5))

kl_1 = dist.kl_divergence(d1, d2)
print("kl 1: {}".format(kl_1))
kl_1.backward()
print(a.grad)

# manual implementation
a.grad.data = torch.zeros(5)
kl_2 = torch.sum(a * (torch.log(a) - np.log(1. / 5)))
print("kl 2: {}".format(kl_2))
kl_2.backward()
print(a.grad)