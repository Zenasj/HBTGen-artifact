import torch
import torch.distributions as td

scale=torch.ones(2,3)
loc=torch.zeros(2,3)
normal = td.Normal(loc=loc, scale=scale)
diag_normal = td.Independent(normal, reinterpreted_batch_ndims=1)
trans_dist = td.TransformedDistribution(diag_normal, transforms=td.AffineTransform(loc=0., scale=2.))
print(td.kl.kl_divergence(trans_dist, trans_dist), "Got incorrect shape")
print(td.kl.kl_divergence(diag_normal, diag_normal), "Correct shape")