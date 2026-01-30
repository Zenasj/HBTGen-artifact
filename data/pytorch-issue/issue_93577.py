import torch

def f(x):
    if x.numel() == 0:
        return x.cos() 
    return x.sin()

torchdynamo.export(f, torch.ones(6, 4), aten_graph=True, tracing_mode="symbolic")