import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(in_features=1, out_features=2)

    def forward(self, v5):
        v6 = torch.neg(v5) # v6: (2, 2, 1, 2, 1) # needed to reproduce
        v3 = self.layer1(v6) # v3: (2, 2, 1, 2, 2)
        return v3

with torch.no_grad(): # needed to reproduce
    model = Model().eval() # remove eval will lead to AssertionError: Fusion only for eval!
    x = torch.rand(2, 2, 1, 2, 1)
    model(x)
    print('==== Eager mode OK! ====')

    compiled = torch.compile(model)
    compiled(x)
    print('==== torch.compile mode OK! ====')