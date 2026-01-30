import torch
import torch.nn as nn

n = torch.nn.LayerNorm(4, eps=1e-5, elementwise_affine=False)
a = torch.randn(5, 4) + 2000  # create random tensor with N=5, L/C=4 and mean=2000, var=1
[[torch.mean(i), torch.var(i, unbiased=False)] for i in n(a)]  # should be near [0, 1]

def expected_LayerNorm(single_layer1d, eps):
    c = single_layer1d
    return (c - torch.mean(c)) / (torch.sqrt(torch.var(c, unbiased=False).add(eps)))

# comparison for one sample of a
print(n(a[2, :]))
print(expected_LayerNorm(a[2, :], 1e-5))

tensor([  22.0000,   72.7500, -285.0000,  190.5000])
tensor([ 0.1249,  0.4133, -1.6235,  1.0852])