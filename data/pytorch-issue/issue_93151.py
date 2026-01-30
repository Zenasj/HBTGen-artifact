import torch._dynamo
from collections import OrderedDict

def f(x):
    y = OrderedDict()
    y['a'] = x
    return y['a']

print(torch._dynamo.export(f, torch.randn(3)))