import torch.nn as nn

import torch

model = torch.nn.Linear(10, 10)
model.qconfig = torch.ao.quantization.qconfig.default_qat_qconfig
torch.ao.quantization.prepare_qat(model, inplace=True)

_ = model(torch.randn(10, 10))
model.apply(torch.ao.quantization.disable_observer)

torch.onnx.export(model, torch.randn(10, 10), "qat-error-example.onnx", opset_version=11)