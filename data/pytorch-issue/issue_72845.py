import torch

dist = torch.distributions

torch_normal = dist.Normal(loc=0.0, scale=1.0)
torch_mixture = dist.MixtureSameFamily(
    dist.Categorical(torch.ones(5,)
    ),
    dist.Normal(torch.randn(5,), torch.rand(5,)),
)

dist.kl_divergence(torch_normal, torch_mixture)