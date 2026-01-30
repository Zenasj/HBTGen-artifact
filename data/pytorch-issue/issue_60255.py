import torch
import torch.nn as nn

input1 = torch.rand(3,3)
input2 = torch.rand(3,3)

class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()

    def forward(self, x1, x2):
        return torch.ops.torch_ipex.my_custom_op(x1,x2)

my_custom_mod = M()

with AMP_ENABLED, torch.no_grad():
    traced= torch.jit.trace(my_custom_mod, [input1, input2])

print(traced.graph)