import torch.nn as nn

from torch.fx import subgraph_rewriter
from torch.fx import symbolic_trace
import torch

class M(torch.nn.Module):
    def forward(self, x, y, z):
        return x + y

def pattern(x, y):
    return x + y

def replacement(x, y):
    return x - y

m = M()
m = symbolic_trace(m)
subgraph_rewriter.replace_pattern(m, pattern, replacement)
print(m)

M()



def forward(self, x, y):
    sub = x - y;  x = y = None
    return sub

def test_subgraph_rewriter_with_unused_args(self):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y

        def pattern(x, y):
            return x + y

        def replacement(x, y):
            return x - y

        def comparison(x1, x2, x3):
            return x1 - x2

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x1 = torch.randn(3, 4)
        x2 = torch.randn(3, 4)
        x3 = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()
        placeholder_nodes = [n for n in traced.graph.nodes if n.op == "placeholder"]
        assert len(placeholder_nodes) == 3

        ref_outs = comparison_fn(x1, x2, x3)
        test_outs = traced.forward(x1, x2, x3)
        self.assertEqual(ref_outs, test_outs)