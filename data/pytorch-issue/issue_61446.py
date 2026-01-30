import torch.nn as nn

import torch
from torch import nn

input_without_grad = torch.rand(())
input_with_grad = input_without_grad.clone().requires_grad_(True)


class ModuleWithBackwardHook(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_backward_hook(lambda *args: None)

    def forward(self, input):
        return input


module_with_backward_hook = ModuleWithBackwardHook()
assert (
    module_with_backward_hook(input_without_grad) is input_without_grad
), "Module with backward hook modifies a input without grad"
assert (
    module_with_backward_hook(input_with_grad) is input_with_grad
), "Module with backward hook modifies a input with grad"

class ModuleWithFullBackwardHook(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_full_backward_hook(lambda *args: None)

    def forward(self, input):
        return input


module_with_full_backward_hook = ModuleWithFullBackwardHook()
assert (
    module_with_full_backward_hook(input_without_grad) is input_without_grad
), "Module with full backward hook modifies a input without grad"
assert (
    module_with_full_backward_hook(input_with_grad) is input_with_grad
), "Module with full backward hook modifies a input with grad"