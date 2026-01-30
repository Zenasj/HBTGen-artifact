import torch

py
@torch.library.custom_op("_reinplacing::sin_cos", mutates_args={"out_sin", "out_cos"})
def sin_cos(x: torch.Tensor, out_sin: torch.Tensor, out_cos: torch.Tensor) -> None:
    out_sin.copy_(x.sin())
    out_cos.copy_(x.cos())

@torch.compile
def f(x):
    out0 = torch.empty_like(x)
    out1 = torch.empty_like(x)
    sin_cos(x, out0, out1)
    return x.clone(), out0, out1

x = torch.randn(3, requires_grad=True)
f(x)