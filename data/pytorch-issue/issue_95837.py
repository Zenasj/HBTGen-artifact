import torch.nn as nn

torch.cuda.amp.autocast

import torch
import torch._dynamo

def fn(x):
    torch._dynamo.graph_break()
    return x.sum() / x.numel()

class MyModule(torch.nn.Module):
    def forward(self, a, b):
        with torch.amp.autocast(device_type="cuda"):
            x = a + b
            y1 = fn(x)
        return y1

module = MyModule()
a = torch.rand((8, 8), dtype=torch.float16, device="cuda")
b = torch.rand((8, 8), dtype=torch.float16, device="cuda")
opt_m = torch._dynamo.optimize("eager")(module)
print(module(a, b).dtype)
print(opt_m(a, b).dtype)

torch.float32
torch.float16

AutocastModeVariable