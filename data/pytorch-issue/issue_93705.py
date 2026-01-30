import torch
import torchdynamo

torchdynamo.config.capture_scalar_outputs = True

def foo(x):
    return x + x.item()

gm, guards = torchdynamo.export(foo, torch.tensor(1), aten_graph=True)

graph, guards = torchdynamo.export(
    foo,
    (torch.randn(1)),
)
print(graph)
make_fx_graph = make_fx(graph)(torch.randn(1))
print(make_fx_graph)