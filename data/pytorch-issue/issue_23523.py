import torch.nn as nn

import torch

class TracedInlineDecision(torch.nn.Module):
    def forward(self, x, flag):
        x = torch.neg(x) # do something random
        @torch.jit.script
        def make_decision(flag, x):
            if flag:
                return x
            else:
                return torch.zeros_like(x)
        return make_decision(flag, x)


decision = TracedInlineDecision()

class Search(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()
        # unfortunately need check_trace because something is broken on master :/
        self.traced = torch.jit.trace(decision, (torch.rand(3, 4), torch.tensor([True], dtype=torch.bool)))


    @torch.jit.script_method
    def forward(self, x):

        for i in range(10):
            x = self.traced(x, torch.tensor([True], dtype=torch.bool))

        x = self.traced(x, torch.tensor([False], dtype=torch.bool))
        return x

s = Search()
print(s.code)