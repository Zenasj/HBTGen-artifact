import torch.nn.functional as F

import torch
from torch.nn import functional as F

C = 8
D = 16  # Cout
K = 10


def f1(features, weights, bias):
    features = features.reshape(N, C, K)
    weights = weights.reshape(N, D, C)
    bias = bias.reshape(N, D, 1)
    out = torch.einsum("nck,ndc->ndk", features, weights) + bias
    return out.reshape(1, N * D, K)


def f2(features, weights, bias):
    return F.conv1d(features, weights, bias, stride=1, padding=0, groups=N)


for N in [3, 0]:
    features = torch.rand(1, N * C, K)
    weights = torch.rand(N * D, C, 1)
    bias = torch.rand(N * D)
    out1 = f1(features, weights, bias)
    out2 = f2(features, weights, bias)
    print(torch.abs(out1.reshape(out2.shape)-out2).max())