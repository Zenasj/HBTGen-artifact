import torch
import torch.nn as nn

class SecondModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        return tensor + tensor


class RecursiveModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.val = torch.nn.Parameter(torch.tensor([2.]))
        self.mod1 = SecondModule()
        self.mod2 = SecondModule()
        self.ln = torch.nn.LayerNorm([1, ])

    def forward(self, input):
        return self.ln(self.mod2(self.val * self.mod1(input)))

mymod = torch.jit.script(RecursiveModule())