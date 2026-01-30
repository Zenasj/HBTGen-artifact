import numpy as np

import copy

import torch
import torch._dynamo as dynamo
import torch._dynamo.config

dynamo.config.repro_after = "dynamo"
dynamo.config.repro_level = 4

def custom_backend(gm: torch.fx.GraphModule, example_inputs):
    gm = copy.deepcopy(gm)
    for node in gm.graph.nodes:
        if len(node.args) > 1:
            node.target = torch.add
            node.args = (node.args[0], 0)
    gm.recompile()
    return gm

inp = torch.ones(5)
inp.requires_grad_(True)

@dynamo.optimize(custom_backend)
def foo(x):
    x = x * x
    return x.sum()

y = foo(inp)
print(y)
y.backward()
print(inp.grad)