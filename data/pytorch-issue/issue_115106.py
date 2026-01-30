import torch.nn as nn

import torch


class SingleOp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.ops.aten.native_batch_norm

    def forward(self, input, weight, bias, running_mean, running_var, training, momentum, eps, **kwargs):
        return self.op(input, weight, bias, running_mean, running_var, training, momentum, eps, **kwargs)


input = torch.randn(5, 5, 5)
weight = torch.randn(5)
bias = torch.randn(5)
running_mean = torch.randn(5)
running_var = torch.randn(5)
training = True
momentum = 0.5
eps = 0.6

model = SingleOp()
output = model(input, weight, bias, running_mean, running_var, training, momentum, eps)
print("Success on torch")

ep = torch.export.export(model, args=(input, weight, bias, running_mean, running_var, training, momentum, eps))
ep.run_decompositions(decomp_table=torch._decomp.decomposition_table)