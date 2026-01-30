import torch
from torch.func import functionalize
from torch.fx.experimental.proxy_tensor import make_fx

y = torch.randn(1, 1)
def f(x):
    return torch.matmul(x, y)

make_fx(
    functionalize(f),
    tracing_mode='symbolic',
    _allow_non_fake_inputs=True,
)(torch.randn(1, 1, 1))