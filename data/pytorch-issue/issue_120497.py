import torch
import torch.distributions as D


@torch.compile
def f():
    d1 = D.Normal(0, 1)
    d2 = D.Normal(2, 1)
    return D.kl_divergence(d1, d2)


print(f())