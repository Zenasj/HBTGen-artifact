import torch
import torch._dynamo.comptime
def f(x):
    y = x.item()
    torch.export.constrain_as_size(y)
    return torch.zeros(y)
print(torch.export.export(f, (torch.tensor([3]),)))