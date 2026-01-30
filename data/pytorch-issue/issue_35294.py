#!python3

import torch
import torch.onnx
import torch.sparse
from torch.nn import Module

class SparseModel(Module):

    def __init__(self):
        super(SparseModel, self).__init__()
        i = [[0, ],
             [0, ]]
        v = [42, ]
        self.s = torch.sparse_coo_tensor(i, v, (1, 1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.s.to_dense() # works if we change the output to "return input" (no usage of sparse tensors)

model = SparseModel()
model.eval()

input_dummy = torch.randn(1, 60, 60, 1, requires_grad=False)
model_out_dummy = model(input_dummy)

model_path = 'model.py'
torch.onnx.export(model, input_dummy, f=model_path, export_params=True, opset_version=14,
                  do_constant_folding=False, input_names=['input'], output_names=['output'])