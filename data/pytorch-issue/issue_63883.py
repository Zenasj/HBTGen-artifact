import torch.nn as nn

import torch
import torch.fx

class MyModule(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)

mod = MyModule()
traced = torch.fx.symbolic_trace(mod)

x = torch.randn(5, 3)
torch.testing.assert_allclose(traced(x), torch.relu(x))

new_instance = traced.__new__(type(traced))
new_instance.__init__(traced, traced.graph)

torch.testing.assert_allclose(new_instance(x), torch.relu(x))
"""
RecursionError: maximum recursion depth exceeded while calling a Python object
"""

import torch
import torch.fx

mod = torch.nn.Module()
graph = torch.fx.Graph()
x = torch.fx.GraphModule(mod, graph)

print(x.__class__.__mro__)
"""
(<class 'torch.fx.graph_module.GraphModule.__new__.<locals>.GraphModuleImpl'>, <class 'torch.fx.graph_module.GraphModule'>, <class 'torch.nn.modules.module.Module'>, <class 'object'>)
"""

new_instance = x.__new__(type(x))
print(new_instance.__class__.__mro__)
"""
(<class 'torch.fx.graph_module.GraphModule.__new__.<locals>.GraphModuleImpl'>, <class 'torch.fx.graph_module.GraphModule.__new__.<locals>.GraphModuleImpl'>, <class 'torch.fx.graph_module.GraphModule'>, <class 'torch.nn.modules.module.Module'>, <class 'object'>)
"""