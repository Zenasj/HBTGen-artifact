import torch.nn as nn

import torch

# define a floating point model where some layers could be statically quantized
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()
        # self.conv = torch.nn.Conv2d(1, 1, 1)
        self.linear = torch.nn.Linear(4, 8)
        self.relu = torch.nn.ReLU()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.linear(x)
        x = self.relu(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

# create a model instance
model_fp32 = M()
model_fp32.eval()

model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
input_fp32 = torch.randn(1, 1, 4)
model_fp32_prepared(input_fp32)

model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

# dynamo export
program = torch.export.export(
    model_int8,
    (input_fp32,),
    # strict=False,  # Error in both cases
)

import torch
from torchao.quantization.quant_api import (
    quantize_,
    int8_dynamic_activation_int8_weight,
    int4_weight_only,
    int8_weight_only
)

# define a floating point model where some layers could be statically quantized
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        # self.conv = torch.nn.Conv2d(1, 1, 1)
        self.linear = torch.nn.Linear(4, 8)
        self.relu = torch.nn.ReLU()
        # DeQuantStub converts tensors from quantized to floating point

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.linear(x)
        x = self.relu(x)
        return x

# create a model instance
model_fp32 = M()
model_fp32.eval()

quantize_(model_fp32, int8_weight_only())

input_fp32 = torch.randn(1, 1, 4)

# dynamo export
program = torch.export.export(
    model_fp32,
    (input_fp32,),
    strict=False,
)

print(program)

import torch
from torchao.quantization.quant_api import (
    quantize_,
    int8_dynamic_activation_int8_weight,
    int4_weight_only,
    int8_weight_only,
    unwrap_tensor_subclass,
)

# define a floating point model where some layers could be statically quantized
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        # self.conv = torch.nn.Conv2d(1, 1, 1)
        self.linear = torch.nn.Linear(4, 8)
        self.relu = torch.nn.ReLU()
        # DeQuantStub converts tensors from quantized to floating point

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.linear(x)
        x = self.relu(x)
        return x

# create a model instance
model = M()
model.eval()

quantize_(model, int8_weight_only())
model = unwrap_tensor_subclass(model)

input_fp32 = torch.randn(1, 1, 4)

# dynamo export
program = torch.export.export(
    model,
    (input_fp32,),
)

print(program)