import torch
import torch._dynamo

def fn(x, y):
    x += y
    return x


fn2 = torch._dynamo.optimize("nnc")(fn)
print(fn(torch.ones(1), torch.ones(1)))
print(fn2(torch.ones(1), torch.ones(1)))