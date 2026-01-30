import torch
import torch.nn as nn

class M(torch.nn.Module):
    def forward(self, x, y, cache):
        m = torch.mul(x, y)
        n = cache.index_copy(0, torch.tensor([0]), m)
        p = torch.ops.aten.copy.default(cache, n)
        q = torch.ops.aten.copy_.default(cache, p)
        u = torch.relu(cache)
        return u  # check the result to ensure cache is updated before relu op

def pattern(self_tensor, src_tensor):
    p = torch.ops.aten.copy.default(self_tensor, src_tensor)
    q = torch.ops.aten.copy_.default(self_tensor, p)
    return q

def replacement(self_tensor, src_tensor):
    q = torch.ops.aten.copy_.default(self_tensor, src_tensor)
    return q

def comparison(x, y, cache):
    m = torch.mul(x, y)
    n = cache.index_copy(0, torch.tensor([0]), m)
    q = torch.ops.aten.copy_.default(cache, n)
    u = torch.relu(cache)
    return u

traced = symbolic_trace(M())
print(traced)
comparison_fn = symbolic_trace(comparison)
print(comparison_fn)

subgraph_rewriter.replace_pattern(traced, pattern, replacement)