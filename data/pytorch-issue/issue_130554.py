import torch
from collections import defaultdict

@torch.compile(fullgraph=True)
def fn(x):
    d = defaultdict(set)
    d['sin'].add(x.cos())
    d['cos'].add(x.sin())
    return d

x = torch.randn(2, 3)
fn(x)

# raises
# torch._dynamo.exc.Unsupported: call_function UserDefinedClassVariable(<class 'collections.defaultdict'>) [BuiltinVariable()] {}