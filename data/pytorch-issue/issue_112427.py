import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.LogSigmoid()
    def forward(self, x):
        return self.m(x)
input = torch.randn(2)

ep = torch.export.export(Model(), args=(input,))
ep.graph_module.graph.print_tabular()
ep.graph_module.print_readable()