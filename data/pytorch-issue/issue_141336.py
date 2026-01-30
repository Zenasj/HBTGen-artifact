import torch.nn as nn

import torch

class UpdateModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.params = torch.zeros((4, 4, 10))

    def forward(self, update, index1, index2):
        copy = self.params.clone()
        copy[index1, torch.tensor([1, 2], dtype=torch.int64), index2] = update
        return copy

model = UpdateModel()

update = (torch.arange(2) + 10).reshape((2,)).to(torch.float32)
index1 = torch.tensor([1, 2]).to(torch.int64)
index2 = torch.tensor([7, 8]).to(torch.int64)
model(update, index1, index2)

ep = torch.export.export(model, (update, index1, index2))
print(ep.graph)
ep.run_decompositions()  # Fails here