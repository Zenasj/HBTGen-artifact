import torch.nn as nn

import torch
from torch_geometric.nn.module_dict import ModuleDict

edge_type = ("a", "to", "b")

class SomeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_dict = ModuleDict({
            edge_type: torch.nn.Linear(1, 1),
        })

    def forward(self, x):
        key = ModuleDict.to_internal_key(edge_type)
        x = self.module_dict[key](x)
        return x

# (1) shows that module has graph breaks
from torch._dynamo.utils import CompileProfiler
with CompileProfiler() as prof:
    model = torch.compile(SomeModel())
    model(torch.randn(1, 1))
    print(prof.report())

# (2) shows that module has NO graph break
model = SomeModel()
explain = torch._dynamo.explain(model)(torch.randn(1, 1))
print(explain)