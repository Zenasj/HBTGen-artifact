def __deepcopy__(m: fx.GraphModule) -> fx.GraphModule:
    fake_mod = nn.Module()
    fake_mod.__dict__ = copy.deepcopy(m.__dict__)
    return fx.GraphModule(fake_mod, m.graph)  # m.graph -> copy.deepcopy(m.graph)

import torch
import torch.nn as nn
import torch.fx as fx
import copy

model = nn.Sequential(nn.Conv2d(1, 1, 1))
gm = fx.symbolic_trace(model)

gm.graph.print_tabular()  # Print original graph.

# Deepcopy and modify (insert node).
dc_gm = copy.deepcopy(gm)
first_node = next(iter(dc_gm.graph.nodes))

with dc_gm.graph.inserting_after(first_node):
    new_node = dc_gm.graph.call_module('inserted', args=(first_node,))
    first_node.replace_all_uses_with(new_node)

gm.graph.print_tabular()  # Print original graph, not modified.