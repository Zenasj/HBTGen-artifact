import torch

def f(x):
    if torch.distributed.is_available():
        return x.cos() + 1
    return 1

torchdynamo.config.dynamic_shapes = True
torchdynamo.export(f, torch.randn(3, 4), aten_graph=True)