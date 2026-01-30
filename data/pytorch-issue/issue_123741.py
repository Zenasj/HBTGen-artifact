import torch

m = torch.library.Library("mylib", "FRAGMENT")
def foo_impl(x: torch.Tensor) -> torch.Tensor:
    return x.sin()

m.define("foo(Tensor x) -> Tensor")
m.impl("foo", foo_impl)

@torch.library.register_fake("mylib::foo")
def _(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape)