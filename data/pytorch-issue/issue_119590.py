import torch
from torch._custom_op.functional import register_functional_op
import torch.utils.checkpoint
from torch.utils.checkpoint import checkpoint, _pt2_selective_checkpoint_context_fn_gen

def custom_policy(mode, func, *args, **kwargs):
    return func in [torch.ops.aten.mm.default]

def selective_checkpointing_context_fn():
    return _pt2_selective_checkpoint_context_fn_gen(custom_policy)

def gn(x, y):
    return torch.selu_(torch.matmul(x, y))

def fn(x, y):
    return torch.utils.checkpoint.checkpoint(
        gn,
        x,
        y,
        use_reentrant=False,
        context_fn=selective_checkpointing_context_fn,
    )

x = torch.arange(16, dtype=torch.float32, requires_grad=True).reshape(4, 4).detach().requires_grad_(True)
y = torch.arange(16, dtype=torch.float32, requires_grad=True).reshape(4, 4).detach().requires_grad_(True)

out1 = gn(x, y)
print(out1)
out1.sum().backward()
print(out1)

out2 = fn(x, y)
print(out2)
# With SAC + eager mode:
# (1) "out" is an activation saved for backward
# (2) selu_() is part of the recompute, which mutates out **again**, during the backward pass!
# Invoking the backward will mutate out!
out2.sum().backward()
print(out2)
# False
print(torch.allclose(out1, out2))