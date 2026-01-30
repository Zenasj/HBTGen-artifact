import torch.nn as nn

py
import torch
torch._C._jit_set_nvfuser_single_node_mode(True)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 1

    def forward(self, input):
        return input.std(self.dim, )

input = torch.rand([4, 4], dtype=torch.float32, device='cuda')

m = M().to('cuda')

print(m(input))

jit_m = torch.jit.script(m)
print(jit_m(input))
print(jit_m(input))

py
import torch
torch._C._jit_set_nvfuser_single_node_mode(True)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 1

    def forward(self, input):
        return input.var(self.dim, )

input = torch.rand([4, 4], dtype=torch.float32, device='cuda')

m = M().to('cuda')

print(m(input))

jit_m = torch.jit.script(m)
print(jit_m(input))
print(jit_m(input))