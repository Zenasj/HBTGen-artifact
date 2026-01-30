import torch.nn as nn

import torch
from torch.nn.modules.lazy import LazyModuleMixin

class module_a(LazyModuleMixin, torch.nn.Module):
    def __init__(self):
        super().__init__()

    def initialize_parameters(self, input):
        return None

    def forward(self, input):
        return input

def hook_function(module, input):
    return input[0] + 1

A = module_a()
A.register_forward_pre_hook(hook_function)
output = A(torch.zeros(2, 2)) # Runtime Error

tensor([[1., 1.],
        [1., 1.]])

for hook in itertools.chain(
                    _global_forward_pre_hooks.values(),
                    self._forward_pre_hooks.values()):
                result = hook(self, input)
                if result is not None:
                    if not isinstance(result, tuple):
                        result = (result,)
                    input = result

for hook in list(itertools.chain(
                    _global_forward_pre_hooks.values(),
                    self._forward_pre_hooks.values())):
                result = hook(self, input)
                if result is not None:
                    if not isinstance(result, tuple):
                        result = (result,)
                    input = result