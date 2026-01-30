import torch
import torch.nn as nn

def f(x):
    return nn.Softmax(dim=-1)(x)

script = torch.jit.script(f)

class ParentModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, x: torch.Tensor):
         return self.softmax(x)

m = ParentModule()
scripted_m = torch.jit.script(m)
print(scripted_m(torch.rand([1, 2, 3])))