import torch.nn as nn

import random
import torch
from torch import nn
from memory_profiler import profile

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(500000, 100)
        self.conv = nn.Conv1d(100, 200, 3)

    def forward(self, x):
        x = self.embed(x)
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        return torch.mean(x)


model = DummyModel()

@profile
def run():
    x = torch.randint(0, 500000, (300, random.randint(100, 3000)))
    model(x)

for i in range(20):
    run()