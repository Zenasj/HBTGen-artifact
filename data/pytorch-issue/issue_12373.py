import torch

distr = torch.distributions.Beta(torch.tensor([1], dtype=torch.float32, device=torch.device('cpu')), torch.tensor([1], dtype=torch.float32, device=torch.device('cpu')))
data = distr.sample((100,))

distr = torch.distributions.Beta(torch.tensor([72], dtype=torch.float32, device=torch.device('cpu')), torch.tensor([15], dtype=torch.float32, device=torch.device('cpu')))
data = distr.sample((1000000,))

# incorrect results
alpha = torch.tensor([0.8453])
beta = torch.tensor([2.3126])
prob = torch.distributions.beta.Beta(
    alpha,
    beta
).rsample((100,))
print(prob)

# correct results
alpha = torch.tensor([0.8453])
beta = torch.tensor([2.3126])
prob = torch.distributions.beta.Beta(
    alpha.item(),
    beta.item()
).rsample((100,))
print(prob)

alpha_beta = torch.tensor([0.8452719449996948, 2.312640905380249])
samples = torch.distributions.Dirichlet(alpha_beta).rsample((10000,))[:, 0]
plt.hist(samples.detach().numpy(), bins=100)
plt.show()

import matplotlib.pyplot as plt

prob = torch.distributions.Dirichlet(torch.tensor([[2.3126, 0.8453], [1, 1]])).rsample((10,))
print(prob)

prob = torch.distributions.Dirichlet(torch.tensor([[2.3126, 0.8453], [1, 1]])).rsample((10000,))
plt.hist(prob[:, 0, 1].detach().numpy(), bins=100)
plt.show()

# this gives incorrect results
prob = torch.distributions.Dirichlet(torch.tensor([[2.3126, 0.8453]])).rsample((10,))
print(prob)

def _dirichlet_sample_nograd(concentration):
    concentration = concentration.clone() # this solves the issue
    probs = torch._standard_gamma(concentration)
    probs /= probs.sum(-1, True)
    return clamp_probs(probs)

def rsample(self, sample_shape=()):
    shape = self._extended_shape(sample_shape)
    # this clone has no effect
    # self.concentration = self.concentration.clone()
    print(self.concentration)
    print(shape)
    concentration = self.concentration.expand(shape)
    # this clone "resolves" the issue
    concentration = concentration.clone()

    if isinstance(concentration, torch.Tensor):
        return _Dirichlet.apply(concentration)
    return _dirichlet_sample_nograd(concentration)

concentration = torch.tensor([[2.3126, 0.8453]]).expand([2, 1, 2])
print(torch._standard_gamma(concentration))
print(torch._standard_gamma(concentration.clone()))

# tensor([[[7.1208e-01, 3.6902e-01]],
#         [[1.1755e-38, 1.1755e-38]]])
# tensor([[[1.4383, 0.5455]],
#         [[3.3665, 0.1593]]])