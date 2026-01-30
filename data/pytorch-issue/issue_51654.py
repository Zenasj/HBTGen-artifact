import torch
import torch.fx

def foo_to_trace(x : 'torch.Tensor') -> 'torch.Tensor':
    return torch.relu(x)

traced = torch.fx.symbolic_trace(foo_to_trace)
"""
  File "str_type.py", line 7, in <module>
    traced = torch.fx.symbolic_trace(foo_to_trace)
  File "pytorch/torch/fx/symbolic_trace.py", line 605, in symbolic_trace
    return GraphModule(tracer.root, graph, name)
  File "pytorch/torch/fx/graph_module.py", line 194, in __init__
    self.graph = graph
  File "pytorch/torch/nn/modules/module.py", line 995, in __setattr__
    object.__setattr__(self, name, value)
  File "pytorch/torch/fx/graph_module.py", line 217, in graph
    self.recompile()
  File "pytorch/torch/fx/graph_module.py", line 295, in recompile
    self._code = self._graph.python_code(root_module='self')
  File "pytorch/torch/fx/graph.py", line 703, in python_code
    emit_node(node)
  File "pytorch/torch/fx/graph.py", line 654, in emit_node
    maybe_type_annotation = '' if node.type is None else f' : {type_repr(node.type)}'
  File "pytorch/torch/fx/graph.py", line 610, in type_repr
    modules_used.add(o.__module__)
AttributeError: 'str' object has no attribute '__module__'
"""