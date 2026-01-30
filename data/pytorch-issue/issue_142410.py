import torch


@torch.library.custom_op("test::f", mutates_args=[])
def f(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


@f.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    ctx = torch.library.get_ctx()
    s = ctx.new_dynamic_size()
    # return torch.empty(s, x.shape[1], device=x.device, dtype=x.dtype)  # This works fine, as the dynamic shape does not influence the strides.
    return torch.empty(x.shape[0], s, device=x.device, dtype=x.dtype)  # This fails.


example = torch.zeros([10, 20], device="cpu")
torch.library.opcheck(f, args=[example])  # This passes.

example = torch.zeros([10, 20], device="cuda")
torch.library.opcheck(f, args=[example])  # This fails.