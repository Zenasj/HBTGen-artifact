import torch
import torch._dynamo as dynamo

@dynamo.optimize()
def func():
    a = torch.rand(1,1,1)
    b = a.repeat(10, 1, 1)
    c = a.repeat_interleave(repeats=10, dim=0)
    return b, c

b, c = func()
torch.equal(b, c)