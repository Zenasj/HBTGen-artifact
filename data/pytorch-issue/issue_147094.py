import torch.nn as nn

import torch

input = torch.randint(0, 8, (5,), dtype=torch.int64)

class BincountDummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        weights = torch.linspace(0, 1, steps=5)
        bc = x.bincount(weights)
        return bc

device = "cpu"
model = BincountDummyModel().to(device)

exported_model = torch.export.export(model, (input,))