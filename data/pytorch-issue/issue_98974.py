import torch
from torch.fx.subgraph_rewriter import replace_pattern_with_filters
from torch.fx.experimental.proxy_tensor import make_fx

def model(x):
    x = x + 2
    x = x[:,:,:]
    return x

def redundant_pattern(input, dim, ind):
    return torch.ops.aten.slice.Tensor(input, dim, 0, ind)
def replacement(input, dim, ind):
    return input

x = torch.randn(2,2,2)
gm = make_fx(model)(x)
gm.graph.print_tabular()
matches = replace_pattern_with_filters(gm, redundant_pattern, replacement, [])
print(matches[2].replacements)  # Expected only [], instead see [x_1, add]