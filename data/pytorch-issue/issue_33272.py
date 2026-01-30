import torch.nn as nn

import torch
from torch import nn

class foo(nn.Module):
    def forward(self, x):
        return x.flatten(start_dim=-2, end_dim=-1)


bar = foo()

out = bar(torch.rand(100, 200, 300))
print(out.shape)

torch.onnx.export(bar, torch.rand(100, 200, 300), 'bar.onnx', verbose=True)

params_dict = torch._C._jit_pass_onnx_constant_fold(graph, params_dict, _export_onnx_opset_version)