import torch
import torch_xla.core.xla_model as xm

@torch.compile(backend="openxla")
def foo(x):
    return x.expand(2, *x.shape)

x = torch.arange(3, device=xm.xla_device())
print(foo(x))