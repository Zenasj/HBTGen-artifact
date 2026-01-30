import torch
import torchdynamo
def f(x: torch.Tensor):
    return x.numel()

torchdynamo.config.capture_scalar_outputs = True
print(torchdynamo.explain(f, torch.ones(1)))
torchdynamo.export(f, torch.ones(1))

import torch
import torchdynamo
def f(x: torch.Tensor):
    return x.numel() + x.sin().numel()

torchdynamo.config.capture_scalar_outputs = True
print(torchdynamo.explain(f, torch.ones(1)))
torchdynamo.export(f, torch.ones(1))

import torch
def f(x: torch.Tensor):
    return x.numel()

torch._dynamo.config.capture_scalar_outputs = True
out =  torch.compile(f, backend="eager", fullgraph=True)(torch.randn(10))