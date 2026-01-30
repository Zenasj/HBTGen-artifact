Inline(*opt_graph);
GRAPH_DEBUG("After Inline, before LowerGradOf\n", *opt_graph);
LowerGradOf(*opt_graph);
GRAPH_DEBUG(
  "After LowerGradOf, before specializeAutogradZero\n", *opt_graph);

import torch

def f(x, y):
  a = x + y
  b = x - y
  c = a * b
  d = c ** 3
  e = d.sum()
  return e

script_f = torch.jit.script(f)
x, h = torch.rand(3, 4), torch.rand(3, 4)
print(script_f(x, h))

script_f = torch.jit.script(f)
print(script_f.graph)