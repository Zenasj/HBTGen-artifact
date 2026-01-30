import torch
from torch.fx.experimental.proxy_tensor import make_fx, wrapper_and_args_for_make_fx

def func(a, b):
    return b.expand([1, a.shape[0], b.shape[-1]])

a = torch.randn(3, 4, device="cuda")
b = torch.randn(4, device="cuda")

class TestMode(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if torch.overrides.resolve_name(func) in ["torch.Tensor.expand"]:
            print(f"TestMode: {func} {args} {kwargs}")
            wrapped, all_args = wrapper_and_args_for_make_fx(func, args, kwargs)
            gm = make_fx(wrapped, tracing_mode="real")(all_args)
            gm.graph.print_tabular()

        return func(*args, **kwargs)

with TestMode():
    gm = make_fx(func, tracing_mode="symbolic")(a, b)

gm.graph.print_tabular()