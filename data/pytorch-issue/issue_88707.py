import torch.nn as nn

import torch
import contextlib
from torch.utils.tensorboard import SummaryWriter

class Model(torch.nn.Module):
    def forward(self, a, b):
        return torch.cat([a, b], dim=1) * 2
        return a + 1, b  # alternatively just returning it also triggers bug

with contextlib.closing(SummaryWriter(log_dir="outputs/test")) as sw:
    sw.add_graph(Model(), [torch.zeros([1, 1]), torch.zeros([1, 0])])