import torch.nn as nn

import torch


class ModelWithAtenFmod(torch.nn.Module):
    def forward(self, x, y):
        return torch.fmod(x, y)


x = torch.randn(3, 4, dtype=torch.float32)
y = torch.randn(3, 4, dtype=torch.float32)
OUT = torch.onnx.export_to_pretty_string(
    ModelWithAtenFmod(), (x, y),
    add_node_names=False,
    do_constant_folding=False,
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN)
print(OUT)