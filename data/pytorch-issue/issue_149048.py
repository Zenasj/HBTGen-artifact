import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def forward(self, x, y):
        return torch.multinomial(x, y.shape[0])

model = Model()
inputs = (
    torch.tensor([[4, 5],[6,7]], dtype=torch.float32),
    torch.tensor([0, 5], dtype=torch.int64),
)
model(*inputs)

DYNAMIC = torch.export.Dim.DYNAMIC
ep = torch.export.export(
    model, inputs, dynamic_shapes={"x": {0: DYNAMIC, 1: DYNAMIC}, "y": {0: DYNAMIC}}
)
print(ep)