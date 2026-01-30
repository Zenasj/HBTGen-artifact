import torch

@torch._dynamo.disable(recursive=False)
def f(x):
    out = SubclassConstructor(x)