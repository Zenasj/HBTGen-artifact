y = some_quantized_tensor
x = some_quantized_tensor
x = bn(conv(x))
x = x + y
x = relu(x)

x = conv_with_add_relu(x, y)

from collections import OrderedDict
import contextlib
import operator
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.ao.quantization.fx.match_utils import (
    MatchAllNode,
)
from torch.ao.quantization.quantize_fx import (
    fuse_fx,
)
from torch.ao.quantization.backend_config import (
    get_qnnpack_backend_config,
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
    get_fbgemm_backend_config
)
from torch.ao.quantization import get_default_qconfig_mapping

import torch.ao.quantization.quantize_fx as qfx

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(3)
        self.iden = nn.Identity()

    def forward(self, x):
        y = x
        y = self.iden(x)
        x = self.conv(x)
        x = self.bn(x)
        x = torch.add(x, y)
        x = self.relu(x)
        return x


m = M().eval()

def fuse_conv_bn_relu(is_qat, relu, add_pattern):
    _, bn_pattern, _ = add_pattern
    bn, conv = bn_pattern
    return conv

def conv_bn_res_relu_root_node_getter(pattern):
    relu, add_pattern = pattern
    _, bn_pattern, _ = add_pattern
    bn, conv = bn_pattern
    return conv

def conv_bn_res_relu_extra_inputs_getter(pattern):
    """ get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    relu, add_pattern = pattern
    _, bn_pattern, extra_input = add_pattern
    bn, conv = bn_pattern
    return [extra_input]
fbgemm_weighted_op_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)
# for pytorch <= 1.13
# conv_bn_res_relu_config = BackendPatternConfig((nn.ReLU, (operator.add, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode))) \
#     .set_fuser_method(fuse_conv_bn_relu) \
#     ._set_root_node_getter(conv_bn_res_relu_root_node_getter) \
#     ._set_extra_inputs_getter(conv_bn_res_relu_extra_inputs_getter)
# for pytorch master
conv_bn_res_relu_config = BackendPatternConfig() \
    ._set_pattern_complex_format((nn.ReLU, (torch.add, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode))) \
    .set_fuser_method(fuse_conv_bn_relu) \
    ._set_root_node_getter(conv_bn_res_relu_root_node_getter) \
    ._set_extra_inputs_getter(conv_bn_res_relu_extra_inputs_getter) \
    .set_dtype_configs(fbgemm_weighted_op_int8_dtype_config)

backend_config = get_fbgemm_backend_config().set_backend_pattern_config(conv_bn_res_relu_config)
# m = fuse_fx(m, backend_config=backend_config)
qmapping = get_default_qconfig_mapping()
prepared_model = qfx.prepare_fx(m, qmapping, (), backend_config=backend_config)
converted_model = qfx.convert_fx(prepared_model, qconfig_mapping=qmapping, backend_config=backend_config)

converted_model.print_readable()

class GraphModule(torch.nn.Module):
    def forward(self, x):
        # No stacktrace found for following nodes
        iden_input_scale_0 = self.iden_input_scale_0
        iden_input_zero_point_0 = self.iden_input_zero_point_0
        quantize_per_tensor = torch.quantize_per_tensor(x, iden_input_scale_0, iden_input_zero_point_0, torch.quint8);  x = iden_input_scale_0 = iden_input_zero_point_0 = None
        
        # File: /home/yy/anaconda3/envs/cpudev/lib/python3.8/site-packages/torch/ao/quantization/fx/tracer.py:103, code: return super().call_module(m, forward, args, kwargs)
        iden = self.iden(quantize_per_tensor)
        
        # No stacktrace found for following nodes
        dequantize_1 = iden.dequantize();  iden = None
        
        # File: /home/yy/anaconda3/envs/cpudev/lib/python3.8/site-packages/torch/ao/quantization/fx/tracer.py:103, code: return super().call_module(m, forward, args, kwargs)
        conv = self.conv(quantize_per_tensor, dequantize_1);  quantize_per_tensor = dequantize_1 = None
        
        # No stacktrace found for following nodes
        dequantize_2 = conv.dequantize();  conv = None
        return dequantize_2