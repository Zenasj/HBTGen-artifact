import torch.nn as nn

import torch
print(torch.__version__)

class M(torch.nn.Module):
    def forward(self, x, y):
        return x + y

m = M()
traced = torch.fx.symbolic_trace(m)

_, _, add, _ = traced.graph.nodes
with traced.graph.inserting_after(add):
    relu = traced.graph.call_function(
        torch.relu, args=(add,))

    add.replace_all_uses_with(relu)

print(relu, relu.all_input_nodes)
traced.recompile()

traced(torch.ones(1), torch.ones(1))

add.replace_all_uses_with(relu, delete_user_cb=lambda node: node is not relu)

import torch
print(torch.__version__)

class M(torch.nn.Module):
    def forward(self, x, y):
        return x + y

m = M()
traced = torch.fx.symbolic_trace(m)

_, _, add, _ = traced.graph.nodes
with traced.graph.inserting_after(add):
    relu = traced.graph.call_function(
        torch.relu, args=(add,))

    add.replace_all_uses_with(relu)

print(relu, relu.all_input_nodes)
traced.recompile()
print(traced)