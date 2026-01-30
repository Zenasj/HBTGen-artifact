import torch
from torch.fx.subgraph_rewriter import replace_pattern_with_filters
from torch.fx.experimental.proxy_tensor import make_fx

def model(x):
    x = (x+2) - 3 + x.T
    return x.unsqueeze(0).squeeze()

def redundant_pattern(input):
    x = torch.ops.aten.unsqueeze.default(input, 0)
    return torch.ops.aten.squeeze.default(x)
def replacement(input):
    return input

x = torch.randn(2,2)
gm = make_fx(model)(x)
gm.graph.print_tabular()
matches = replace_pattern_with_filters(gm, redundant_pattern, replacement, [])
print(matches[0].replacements)  # Expected only [add_2], instead see all parents of add_2

for ret_node in copied_returning_nodes:
            if ret_node in match.placeholder_nodes:
                replacement_nodes.append(ret_node)
            else:
                get_replacement_nodes(ret_node)