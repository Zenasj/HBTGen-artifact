from typing import List

import torch
import torch._dynamo as dynamo
import operator

def miscompile_div_as_mul_backend(gm: torch.fx.GraphModule,
                                  example_inputs: List[torch.Tensor]):
    for node in gm.graph.nodes:
        if node.op == "call_function":
            if node.target == operator.truediv:
                node.target = operator.mul
    gm.recompile()
    return gm

@dynamo.optimize(miscompile_div_as_mul_backend)
def f(x, y):
    a = x * y
    b = x + y
    c = x / y
    return a, b, c


args = (torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.]))
print(f(*args))

def forward(self, x : torch.Tensor, y : torch.Tensor):
        mul = x * y
        add = x + y
        truediv = x / y;  x = y = None
        return (mul, add, truediv)