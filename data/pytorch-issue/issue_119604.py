import torch

node: torch.fx.Node = ...
arg1: torch.fx.Argument = node.args[0]
arg2: torch.fx.Argument = node.args[1]
a, b = arg1.meta, arg2.meta
# do something with a & b

def get_arg(arg_type: Type[T], i: int) -> T:
  assert(isinstance(self.args[i], arg_type))
  return self.args[i]