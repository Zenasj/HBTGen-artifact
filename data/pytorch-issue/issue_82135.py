import torch.nn as nn

from enum import Enum


class Foo(Enum):
  A = 1
  B = 2

import torch.fx

def leaf_fn(proxy, enum_val):
  return proxy + enum_val

def foo(x):
  return leaf_fn(x, Foo.A)

traced = torch.fx.symbolic_trace(foo)
print(traced)

import torch.fx

g = torch.fx.Graph()

x = g.placeholder('x')
fn = g.call_function(leaf_fn, (x, Foo.A))
g.output(fn)

gm = torch.fx.GraphModule(torch.nn.Module(), g)