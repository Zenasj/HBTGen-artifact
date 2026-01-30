import torch.nn as nn

# Submodule called by outer module (outer module is `torch.compile`d)
def forward(self, x):
    # the graph break below results in an additional graph break here
    ...
    torch._dynamo.graph_break()
    ...
    # and here
    return x

import torch


class HasGraphBreak(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 5)
        self.linear2 = torch.nn.Linear(5, 5)

    def forward(self, x):
        x = self.linear1(x)
        torch._dynamo.graph_break()
        return self.linear2(x)


class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.submodule = HasGraphBreak()

    def forward(self, x):
        x = torch.relu(x)
        x = self.submodule(x)
        return torch.relu(x)


def custom_backend(gm, _ei):
    _ = gm.print_readable()
    return gm.forward


m = torch.compile(Mod(), backend=custom_backend)

x = torch.randn(3, 5)

_ = m(x)

class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_
        
        # File: graph_breaks.py:27, code: x = torch.relu(x)
        x = torch.relu(l_x_);  l_x_ = None
        return (x,)
        
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_
        
        # File: graph_breaks.py:11, code: x = self.linear1(x)
        x = self.L__self___linear1(l_x_);  l_x_ = None
        return (x,)
        
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_
        
        # File: graph_breaks.py:13, code: return self.linear2(x)
        l__self___linear2 = self.L__self___linear2(l_x_);  l_x_ = None
        return (l__self___linear2,)
        
class GraphModule(torch.nn.Module):
    def forward(self, L_stack0_ : torch.Tensor):
        x = L_stack0_
        
        # File: graph_breaks.py:29, code: return torch.relu(x)
        relu = torch.relu(x);  x = None
        return (relu,)