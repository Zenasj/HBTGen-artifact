import torch.nn as nn

import torch
from torch.fx import symbolic_trace, subgraph_rewriter

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w1, w2):
        m1 = torch.cat([w1, w2]).sum()
        m2 = torch.cat([w1, w2]).sum()
        return x + torch.max(m1) + torch.max(m2)

model = M()

traced_module = symbolic_trace(model)
dummy_in = torch.rand(2,10)
exported_module = torch.export.export(model, (dummy_in, dummy_in, dummy_in))

def pattern(w1, w2):
    return torch.cat([w1, w2]).sum()

def replacement(w1, w2):
    return torch.stack([w1, w2])

# Doesn't replace anything for exported graph
subgraph_rewriter.replace_pattern(exported_module.graph_module, pattern, replacement)

# Pattern is replaced in the symbolic traced graph.
subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)

# Symbolic trace the exported graph, the pattern can be replaced
export_traced = symbolic_trace(model)
subgraph_rewriter.replace_pattern(export_traced, pattern, replacement)

pattern_ep = torch.export.export(pattern, (dummy_in, dummy_in))
subgraph_rewriter.replace_pattern(exported_module.graph_module, pattern_ep.graph_module, replacement_ep.graph_module)