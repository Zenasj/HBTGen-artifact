import torch
@torch.library.custom_op("mylib::clone", mutates_args={})
def f(x: torch.Tensor) -> torch.Tensor:
    return x.clone()
def f_fake(x):
    return torch.empty_like(x)
def backward(ctx, grad):
    ctx.x.zero_()
    return grad
def setup_context(ctx, inputs, output):
    x, = inputs
    ctx.x = x
f.register_fake(f_fake)
f.register_autograd(backward, setup_context=setup_context)
x = torch.randn(3, requires_grad=True)
y = f(x)
y.sum().backward()
def fn(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.mylib.clone(x)
print(x.grad)  # ones
print(x)  # zeros
torch.compile(fn, backend="aot_eager", fullgraph=True)(x)