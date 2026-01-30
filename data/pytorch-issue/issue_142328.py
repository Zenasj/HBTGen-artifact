py
import math

import torch
import torch.nn as nn


def newtonschulz5(G, steps: int, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


@torch.compile
def scaled_newton_schulz(G, steps: int):
    shape = G.shape
    dtype = G.dtype
    G = G.reshape(shape[0], -1)
    G = newtonschulz5(G, steps)
    G = G.reshape(shape).type(dtype)
    G = G * math.sqrt(max(1, shape[0] / G[0].numel()))
    return G


model = nn.Sequential(
    nn.Linear(16, 16, bias=False),
    nn.Linear(16, 32, bias=False),
).cuda()


loss = model(torch.randn(4, 16, device="cuda")).sum()
loss.backward()

scaled_newton_schulz(model[0].weight.grad, 6)
scaled_newton_schulz(model[1].weight.grad, 6)