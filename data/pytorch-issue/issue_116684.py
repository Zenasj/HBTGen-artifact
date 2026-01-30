import torch.nn as nn

py
import torch
import torch.nn.functional as F

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def forward(self, hidden_states):
        return F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")

ep = torch.export.export(NeuralNetwork(), (torch.zeros(1, 3, 20, 20),))

ep = ep.run_decompositions(decomp_table={})

decomp = torch.export.default_decompositions()
del decomp[torch.ops.aten.upsample_nearesr2d.vec] 
ep = torch.export.export().run_decompositions(decomp)