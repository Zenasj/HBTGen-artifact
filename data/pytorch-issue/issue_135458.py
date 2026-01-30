import torch.nn as nn

import torch
from torch._functorch.aot_autograd import aot_module_simplified

torch._dynamo.config.cache_size_limit = 2
torch._dynamo.config.accumulated_cache_size_limit = 512

class MyGraphModule(torch.nn.Module):
    def __init__(self, graph_module):
        super().__init__()
        self._graph_module = graph_module

    def forward(self, *args):
      print("Calling MyGraphModule", id(self))
      return self._graph_module(*args)

def my_backend(gm, sample_inputs):
    def my_compiler(gm, sample_inputs):
        print("Compiling graph:", gm.code)
        mygm = MyGraphModule(gm)
        return mygm.forward
 
    return aot_module_simplified(
        gm,
        sample_inputs,
        fw_compiler=my_compiler
    )
        

def f(x, g):
    x = x * x
    x = x + 1
    out = torch.utils.checkpoint.checkpoint(g, x)
    return out

def g1(x):
    w = x.sin()
    z = w.sin()
    return z

def g2(x):
    w = x.sin()
    z = w.sin()
    return z

def g3(x):
    w = x.sin()
    z = w.sin()
    return z


f = torch._dynamo.optimize(backend=my_backend)(f)

x = torch.ones(2, requires_grad = True)

print("Calling g1", f)
y = f(x, g1)
print("Calling g2", f)
y = f(x, g2)
print("Calling g3", f)
y = f(x, g3)
print("Calling g1", f)
y = f(x, g1)
print("Calling g2", f)
y = f(x, g2)
print("Calling g3", f)
y = f(x, g3)