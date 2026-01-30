import torch.nn as nn

import torch

class PG(torch.nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis
    def forward(self, indices, data):
        b = data.shape[self.axis]
        return indices + b

pg = PG(0)
sample_input = (torch.IntTensor(), torch.IntTensor(),)
torch.onnx.export(pg, sample_input, 'pg.onnx', verbose=True, opset_version=11,
				input_names=['indices', 'data'], output_names=['pos_indices'], do_constant_folding=True)