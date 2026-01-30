import torch
import torch._dynamo

def fn(x):
    x = x + 1
    print("BREAK")
    x = x + 1
    return x


out = torch._dynamo.explain(fn, torch.randn(10))
print(out)