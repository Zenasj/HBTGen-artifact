import torch
import torch._dynamo

torch._dynamo.config.print_graph_breaks = True

def f(x):
    y = x + 1
    torch._dynamo.graph_break()
    z = y + 1
    return z

def f2(x):
    y = x + 1
    torch._dynamo.graph_break()
    z = y + 1
    return z

@torch.compile()
def g(x):
    return f(x) + f(x) + f2(x)

g(torch.randn(2))

import torch._dynamo.utils
print(torch._dynamo.utils.counters['graph_break'])