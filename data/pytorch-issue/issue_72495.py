import torch

# Construct a Gaussian copula from a multivariate normal.
base_dist = MultivariateNormal(
    loc=torch.zeros(2),
    scale_tril=LKJCholesky(2).sample(),
)
transform = CumulativeDistributionTransform(Normal(0, 1))
copula = TransformedDistribution(base_dist, [transform])

transforms = [
    CumulativeDistributionTransform(Normal(0, 1)),
    CumulativeDistributionTransform(Weibull(4, 2)).inv,
]
wrapped_copula = TransformedDistribution(base_dist, transforms)