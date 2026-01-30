import torch
import torch._dynamo
from torch.onnx._internal import exporter

def func(x):
    return x + torch.full(x.shape, torch.tensor(torch.finfo(x.dtype).min))

options = exporter.ResolvedExportOptions(None)

x = torch.randn(3, 4)
gm, _ = torch._dynamo.export(func, x, aten_graph=True, decomposition_table=options.decomposition_table)
gm.print_readable()

import torch
import torch._dynamo
from torch.onnx._internal import exporter
from torch._dynamo import config


def func(x):
    return x + torch.full(x.shape, torch.tensor(torch.finfo(x.dtype).min))


x = torch.randn(3, 4)
# Step 1: Set config.capture_scalar_outputs = True
config.capture_scalar_outputs = True
# Step 2: Enable dynamic shapes
torch.onnx.dynamo_export(
    func, x, export_options=torch.onnx.ExportOptions(dynamic_shapes=True)
)