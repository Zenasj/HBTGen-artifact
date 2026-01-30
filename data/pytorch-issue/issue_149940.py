import torch.nn as nn

import torch

norm_model = torch.nn.LayerNorm(256)
norm_x = torch.rand(950, 1, 256)

torch.onnx.export(
    norm_model,
    norm_x,
    'norm_test_dynamo_false.onnx',
    input_names=['input'],
    output_names=['output'],
    opset_version=17,
    dynamo=False
)

torch.onnx.export(
    norm_model,
    norm_x,
    'norm_test_dynamo_true.onnx',
    input_names=['input'],
    output_names=['output'],
    opset_version=17,
    dynamo=True
)