import torch._dynamo

def f(x):
    y = dict([('a', x)])
    return y['a']

print(torch._dynamo.export(f, torch.randn(3)))