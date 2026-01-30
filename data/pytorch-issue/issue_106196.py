import torch.nn as nn

import torch

torch.manual_seed(1337)

B, T, C = 1, 2, 3
input = torch.randn(B, T, C)
c_attn = torch.nn.Linear(C, 4*C, bias=True)

with torch.no_grad():
    output = c_attn(input)
    sliced_output = c_attn(input[:,[-1],:])

assert torch.equal(output[:,[-1],:], sliced_output) == True # Note: the bug is here, we expect a True return