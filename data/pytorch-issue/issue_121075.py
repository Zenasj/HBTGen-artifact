import torch

@torch.compile(backend="openxla")
def foo(x):
    new_x = x.new(*x.size())

    # new_x.device() == "xla"
    # x.device() == "xla:0"

    return new_x + x

a = torch.arange(10)
foo(a.to(xm.xla_device()))