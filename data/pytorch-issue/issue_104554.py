import torch
import torch.nn as nn

class ModWithDeadCode(torch.nn.Module):
            def forward(self, x):
                output = x * 2 # we want this
                dead_line = x + 2 # this is dead
                return output

class GraphModule(torch.nn.Module):
    def forward(self, x):
        # No stacktrace found for following nodes
        submod_2 = self.submod_2(x)
        submod_1 = self.submod_1(x);  x = None
        return submod_1

    class GraphModule(torch.nn.Module):
        def forward(self, x):
            # No stacktrace found for following nodes
            add = x + 2;  x = None
            return None

    class GraphModule(torch.nn.Module):
        def forward(self, x):
            # No stacktrace found for following nodes
            mul = x * 2;  x = None
            return mul