import torch
import torch._dynamo as dynamo
from torch._dynamo.optimizations.training import aot_autograd
import functorch

@functorch.compile.make_boxed_compiler
def my_backend(gm, example_inputs):
    gm.graph.print_tabular()
    return gm

@dynamo.optimize(aot_autograd(fw_compiler=my_backend))
def f(x):
    x[0].relu_()
    return x

m1 = -torch.ones(2, 5)
result = f(m1)
print("m1", m1)
print("result", result)
print("m1 is result:", m1 is result)

import torch
import torch._dynamo as dynamo
import functorch
from functorch.experimental import functionalize

def my_backend(gm, example_inputs):
    # Functionalize and convert to dispatcher ops
    gm = functorch.make_fx(functionalize(gm, remove="mutations_and_views"))(*example_inputs)
    gm.graph.print_tabular()
    return gm

@dynamo.optimize(my_backend)
def f(x):
    x[0].relu_()
    return x

m1 = -torch.ones(2, 5)
result = f(m1)
print("m1", m1)
print("result", result)
print("m1 is result:", m1 is result)