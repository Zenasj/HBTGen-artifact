from torch._dynamo import optimize
import torch

@optimize("inductor")
def fn(x, y):
    a = torch.cos(x)
    b = torch.sin(y)
    return a + b


a = fn(torch.Tensor(1),torch.Tensor(1))
print(a)