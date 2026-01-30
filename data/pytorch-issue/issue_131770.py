py
import torch
from torch.utils.flop_counter import FlopCounterMode, register_flop_formula

@torch.library.custom_op("mylib::foo", mutates_args=())
def foo(x: torch.Tensor) -> torch.Tensor:
    return x.sin()

@register_flop_formula(torch.ops.mylib.foo)
def formula(*args, **kwargs):
    raise RuntimeError("called")


x = torch.randn(3)
with FlopCounterMode():
    foo(x)