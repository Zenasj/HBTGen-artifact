import torch

def create():
    torch._dynamo.graph_break()
    return {0: torch.randn(3, 3)}

def fn():
    return {**create()}

opt_fn = torch.compile(backend="eager")(fn)
opt_fn()