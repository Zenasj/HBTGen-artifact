import torch
import torch._dynamo as dynamo

@dynamo.optimize()
def f():
    return torch.ops.aten.len("hello")

f()