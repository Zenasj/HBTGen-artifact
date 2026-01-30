import torch
import torch.nn as nn

class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            # net1, bias are all unused params.
            self.net1 = nn.Linear(10, 5, bias=False)
            self.bias = nn.Parameter(torch.zeros(5))
            self.net2 = nn.Linear(10, 5)

        def forward(self, x):
            return self.net2(x).sum()