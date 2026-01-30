import torch
@torch.library.custom_op("mylib::numpy_mul", mutates_args=())
def numpy_add(x: Tensor, y: float) -> Tensor:
    x_np = x.numpy(force=True)
    z_np = x_np + y
    return torch.from_numpy(z_np).to(x.device)
@numpy_sin.register_fake
def _(x, y):
    return torch.empty_like(x)

sample_inputs = [
    (torch.randn(3), 3.14),
]
for args in sample_inputs:
    torch.library.opcheck(foo, args)