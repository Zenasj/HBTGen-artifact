import torch
import torch.nn as nn
import torch.fx

@torch.jit.script
class Helper:
    @torch.jit.export
    def hello(self):
        print("hello")

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.helper = Helper()

    def forward(inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1,1,1)


m = M()
m.helper.hello()

m.eval()

scripted = torch.jit.script(m)
frozen = torch.jit.freeze(scripted, preserved_attrs=['helper'])
torch.jit.save(frozen, 'frozen-test.pt')

test = torch.jit.load('frozen-test.pt')
test.helper.hello()

import torch

m = torch.jit.load('frozen-test.pt')

m.helper.hello()