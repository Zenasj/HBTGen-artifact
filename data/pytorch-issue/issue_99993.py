import torch

def foo(x, y):
    return x + y

x = torch.rand(5)
y = torch.rand(5)
gm, _ = dynamo.export(
    foo,
    x,
    y,
    constraints=[
        dynamic_dim(x, 0),
        dynamic_dim(y, 0),
    ],
    aten_graph=True,
    tracing_mode="symbolic",
    assume_static_by_default=True,
)

gm(torch.rand(4), torch.rand(6))