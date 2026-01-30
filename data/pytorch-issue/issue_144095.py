import torch

@torch.library.custom_op("test::f", mutates_args=[])
def f(x: torch.Tensor) -> torch.Tensor:
    return x.new_zeros(512, 1)

@f.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    ctx = torch.library.get_ctx()
    s = ctx.new_dynamic_size()
    return torch.empty(s, 1, device=x.device, dtype=x.dtype)




@torch.library.custom_op("test::g", mutates_args=[])
def g(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, x[0].unsqueeze(-1)])

@g.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, x[0].unsqueeze(-1)])




@torch.library.custom_op("test::h_mutate", mutates_args=['x'])
def h_mutate(x: torch.Tensor) -> None:
    x.mul_(2)



@torch.library.custom_op("test::i", mutates_args=[])
def i(x: torch.Tensor, sz: int) -> torch.Tensor:
    return torch.ones(sz, 1, dtype=x.dtype, device=x.device)

@i.register_fake
def _(x: torch.Tensor, sz: int) -> torch.Tensor:
    return torch.empty(sz, 1, dtype=x.dtype, device=x.device)



@torch.library.custom_op("test::j", mutates_args=[])
def j(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + 1

@j.register_fake
def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sz1 = x.shape[0] - 1
    sz2 = y.numel()
    torch._check(sz1 == sz2)
    # make this a reduction so partitioner bans recompute of it
    return x.sum()


torch._dynamo.config.capture_dynamic_output_shape_ops = True
import torch._functorch.config as functorch_config
functorch_config.activation_memory_budget = 0.5

@torch.compile(backend="aot_eager_decomp_partition")
def f(x, param):
    y = torch.ops.test.f(x)
    z = torch.ops.test.g(y)
    z2 = torch.ops.test.i(x, z.shape[0] - 1)
    z2 = torch.ops.test.j(z, z2)
    return torch.matmul(x, param).sin() * z2.sum()

x = torch.randn(512, 512, device='cuda')
param = torch.randn(512, 512, device='cuda', requires_grad=True)
out = f(x, param)