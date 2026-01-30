import torch

def foo(t, y):
    out_1 = torch.ones(1)
    return torch.add(t, y, out=out_1)

g = make_fx(functionalize(foo))(torch.tensor([1]), torch.tensor([1]))
print(g.code)
out1 = functionalize(foo)(torch.tensor([1]), torch.tensor([1]))
out2 = foo(torch.tensor([1]), torch.tensor([1]))
print(out1 == out2)