import torch.nn as nn

import torch

class TensorAssignModel(torch.nn.Module):
    def __init__(self):
        super(TensorAssignModel, self).__init__()

    def forward(self, x):
        x[2] = 9
        return x

x = torch.randn(5, 4)
torch_m = TensorAssignModel()
torch_m = torch.jit.trace(torch_m, x.clone())

graph = torch_m.forward.graph
print(graph)    # This looks good.

torch._C._jit_pass_remove_inplace_ops(graph)
print(graph)    # This looks reasonable too.

torch._C._jit_pass_dce(graph)
print(graph)    # The graph is now empty!