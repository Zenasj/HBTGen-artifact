import torch

def f(x):
    return (x + x).to(torch.int16)

x = torch.tensor(128, dtype=torch.uint8)
torch._dynamo.utils.same(f(x), torch.compile(f)(x))