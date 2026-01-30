import torch

dist = torch.distributions

torch_normal = dist.Normal(loc=0.0, scale=1.0)
torch_mixture = dist.MixtureSameFamily(
    dist.Categorical(torch.ones(5,)
    ),
    dist.Normal(torch.randn(5,), torch.rand(5,)),
)

dist.kl_divergence(torch_normal, torch_mixture)

import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions

tf_normal = tfd.Normal(loc=0.0, scale=1.0)
tf_mixture = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
    components_distribution=tfd.Normal(
        loc=[-0.2, 1], scale=[0.1, 0.5]  # One for each component.
    ),
) 

tfd.kl_divergence(tf_normal, tf_mixture)