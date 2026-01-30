import torch.nn as nn
class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.module_list = nn.ModuleList(
            [nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 1)]
        )

    def forward(self, batch):
        for i in self.module_list[1:4]:
            pass
        return batch
model = Model()
out = model(torch.randn(1, 1))

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_list = nn.ModuleList(
            [nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 1)]
        )

    def forward(self, batch):
        for i in self.module_list[1:4]:
            print(i)
        return batch

model = Model()
out = model(torch.randn(1, 1))
print(out.shape)