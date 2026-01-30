import torch
import torch.fx as fx
import torch.nn as nn


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MyOtherModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(24, 32, 3)

    def forward(self, x):
        x = nn.functional.relu(self.conv(x))


def insert_after(mod: fx.GraphModule, from_idx: int, insertme: fx.GraphModule):
    mod.add_submodule('inserted', insertme)
    cutoff_node: fx.Node = list(mod.graph.nodes)[from_idx]
    next_node: fx.Node = list(mod.graph.nodes)[from_idx+1]

    with mod.graph.inserting_after(cutoff_node):
        new_node = model.graph.call_module('inserted', (cutoff_node,), {})

    next_node.replace_input_with(cutoff_node, new_node)

    mod.delete_all_unused_submodules()
    mod.graph.eliminate_dead_code()
    mod.recompile()
    mod.graph.lint()

    return mod


mod = fx.symbolic_trace(MyModule())
other = fx.symbolic_trace(MyOtherModule())
new_mod = insert_after(mod, 0, other)