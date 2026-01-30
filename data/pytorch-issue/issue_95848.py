import torch

torch._dynamo.config.log_level = logging.WARNING
torch._dynamo.config.print_shape_guards = True
torch._dynamo.config.print_specializations = True

def foo(x, y, z):
    if (
        2 * x.shape[0] == y.shape[0] + z.shape[0]
        and x.shape[0] == z.shape[0] + 1
        and 2 * z.shape[0] == y.shape[0] + 4
    ):
        return 2 * y
    else:
        return 2 + y


m, gs = dynamo.export(
    foo,
    torch.rand(7),
    torch.rand(8),
    torch.rand(6),
    aten_graph=True,
    tracing_mode="symbolic",
)