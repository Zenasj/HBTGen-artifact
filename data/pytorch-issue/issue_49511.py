import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(
        self, dropout: float=0.0
    ):
        super(Model, self).__init__()
        self.do = torch.nn.Dropout(dropout)

    def forward(self, input0: torch.Tensor):
        return self.do(input0)

# this works:
# dropout_rate: float = 0.0

# this doesn't:
dropout_rate: float = 0

model = Model(dropout_rate)
scripted = torch.jit.script(model)