import torch
import torch.fx

@torch.fx.wrap
def foo(x):
    if x.ndim > 1:
        return torch.relu(x)
    else:
        return torch.neg(x)


def to_trace(x):
    return foo(x)


traced = torch.fx.symbolic_trace(to_trace)

print(traced.code)
"""
Code emitted in notebook:
def forward(self, x):
    foo = __main___foo(x);  x = None
    return foo
"""
"""
Code emitted in Python script:
torch.fx._symbolic_trace.wrap("__main___foo")

def forward(self, x):
    foo = __main___foo(x);  x = None
    return foo
"""

for node in traced.graph.nodes:
  print(node.meta)
"""
{}
{}
{}
"""

for node in traced.graph.nodes:
    print(node.meta)
"""
{}
{'is_wrapped': True}
{}
"""