import torch

torch._dynamo.config.capture_dynamic_output_shape_ops = True

from torch._functorch import config
config.ban_recompute_not_in_allowlist = False

@torch.compile(backend="aot_eager")
def f(x):
    y = x.nonzero()
    tmp = torch.ones_like(y)
    return x.sum() + tmp.sum()

x = torch.ones(4, requires_grad=True)
out = f(x)