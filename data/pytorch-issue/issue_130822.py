import torch.nn as nn

import torch
from torch.ao.quantization import get_default_qconfig_mapping, QConfig, QConfigMapping
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torchvision import models

from fx_user_model_fp32.user_model import user_model
from export_dynamicLib_func import DataProcesser, ValifyDynamicLib
from torch.ao.quantization.observer import (
    default_observer, 
    default_weight_observer,
    MovingAveragePerChannelMinMaxObserver,
)

def create_custom_qconfig_mapping():
    # 创建默认的 qconfig
    qconfig = QConfig(activation=default_observer, weight=default_weight_observer)
    # 创建 QConfigMapping 并设置为只对 nn.Linear 进行量化
    qconfig_mapping = QConfigMapping().set_object_type(torch.nn.Linear, qconfig)
    return qconfig_mapping


qconfig_mapping = create_custom_qconfig_mapping()

# model_fp32 = MyModel().eval()
model_fp32 = user_model().eval().to('cpu')
_, inputs_list = DataProcesser.read_inputs_dict(device="cpu")
prepared_model = prepare_fx(model_fp32, qconfig_mapping, example_inputs=tuple(inputs_list))

# Calibration
# calibration_data_loader = ...
# for x in calibration_data_loader:
#     x = calibration_data_loader()
#     prepared_model(x)

# Convert to quantized model
quantized_model = convert_fx(prepared_model)

_, inputs_list = DataProcesser.read_inputs_dict(device="cpu")
y = quantized_model(*inputs_list)

import os
from int8_quantization import Int8Quantization
dynamicLib_path = torch._export.aot_compile(
    quantized_model,
    args = tuple(inputs_list),
    dynamic_shapes = Int8Quantization._get_dynamic_shapes(inputs_list),
    options={
            "aot_inductor.output_path": os.path.join('dynamicLib', "quantized_x86_model_cpu_fp32.so"), 
            "max_autotune": True
            },
)

converted_model = convert_pt2e(prepared_model)

optimized_model = torch.compile(quantized_model)

with torch.no_grad():
        optimized_model = torch.compile(quantized_model)
        optimized_model(*inputs_list)
        dynamicLib_path = torch._export.aot_compile(
            optimized_model,
            args = tuple(inputs_list),
            dynamic_shapes = dynamic_shapes,
            options={
                    "aot_inductor.output_path": os.path.join('dynamicLib', "quantized_model_cpu_fp32.so"), 
                    "max_autotune": True
                    },
        )

with torch.no_grad():
        # Optional: using the C++ wrapper instead of default Python wrapper
        optimized_model = torch.compile(quantized_model)
        optimized_model(*inputs_list)