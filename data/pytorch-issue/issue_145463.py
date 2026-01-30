import torch.nn as nn

import torch
import torch._dynamo.config
from torch.utils.checkpoint import _is_compiling

@torch.compile()
class Y(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(-1, self._N_input, 192)

@torch.compile()
class M(torch.nn.Module):
    def __init__(self, input):
        super().__init__()
        self._N_input = input.size()[0]

    def forward(self, x, z):
        # input is [B, n * d]
        x = x * 2
        x = x * 5
        x = x * 3
        y = Y()
        y._N_input = self._N_input
        x = y(x)  # [B, n, d]
        x = x * 20
        x = x * 30
        x = x * 43
        return x

# Input tensors
x = torch.randn(5, 3210, 192)  # Shape [B=5, n*d=3210]
num_inputs = torch.randn(3210)  # Shape [3210]

# Mark dimensions as dynamic
torch._dynamo.mark_dynamic(x, 1)
torch._dynamo.mark_dynamic(num_inputs, 0)

# Initialize model
m = M(num_inputs)

# Compile the model
compiled_m = torch.compile(m)

# Forward pass
output1 = compiled_m(x, 192)
print(output1)