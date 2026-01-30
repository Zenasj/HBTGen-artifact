import torch.nn as nn

import torch
import logging

@torch._dynamo.disable
def break_gn(x):
    return torch.sin(x)

def gn(x0, x):
    return x0 * break_gn(x)

class MyMod(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch._dynamo.disable(recursive=False)
    def forward(self, input):
        input = torch.sin(input)
        x = input
        x = gn(input, input)
        x = gn(input, x)
        x = gn(input, x)
        return x


torch.cuda.memory._record_memory_history(stacks="python")

mod = MyMod().cuda()
fn = torch.compile(mod, backend="eager")
x = torch.randn(10, 10).cuda()
for _ in range(400):
    fn(x)

torch.cuda.memory._dump_snapshot("my_snapshot.pickle")