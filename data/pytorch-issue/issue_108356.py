import torch.nn as nn

import torch
from functorch.experimental.control_flow import cond

class MySubModule(torch.nn.Module):
    def foo(self, x):
        return x.cos()

    def forward(self, x):
        return self.foo(x)


class CondBranchClassMethod(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.subm = MySubModule()

    def bar(self, x):
        return x.sin()

    def forward(self, x):
        return cond(x.shape[0] <= 2, self.subm.forward, self.bar, [x])


from torch._export import capture_pre_autograd_graph

example_inputs = (torch.randn(1, 3, 3, 3),)
m = CondBranchClassMethod()
m.eval()
gm = capture_pre_autograd_graph(m, example_inputs)
print(gm)

# source_fn for original cond op, getattr submodule op are all cond op
for n in gm.graph.nodes:
    print("n:", n.format_node(), n.meta)

print("\n\n\n")
# source_fn for submodule nodes are all cond op
# Expected: ideally this should be the real ops, e.g. torch.sin, aten.cos, etc
for n in gm.submodule_0.graph.nodes:
    print("n:", n.format_node(), n.meta)