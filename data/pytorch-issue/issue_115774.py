import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cache = torch.tensor([0,0])

    @torch.compile(backend="eager")
    def forward(self, x, pos):
        self.cache[pos] = x
        return self.cache

model = Model()

print(model(torch.tensor([2]), torch.tensor([0])))
# resetting cache/buffer
model.cache = torch.tensor([0,0])
# expect [0, 1], but get [2, 1]
print(model(torch.tensor([1]),torch.tensor([1])))