import torch.nn as nn

torch.randn

import torch

from torch.utils.tensorboard import SummaryWriter

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

    def forward(self, x):
        return torch.randn(10, 10)

model = SimpleModel()
dummy_input = (torch.zeros(1, 2, 3),)

with SummaryWriter(comment='randModel') as w:
    w.add_graph(model, dummy_input, True)