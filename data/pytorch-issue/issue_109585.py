import torch.nn as nn

# @parse_args("v", "v", "f", "f")
# def convert_quant_conv_transpose2d(g, q_input, q_weight, output_scale, output_zero_point):
#     inp, input_scale, _, _ = dequantize_helper(g, q_input)
#     unpacked_inputs = _unpack_param(q_weight)#2 6 51 52
#     output = opset9.conv2d(g, input, weight, bias, stride, padding, dilation, groups)
#     return quantize_helper(g, output, op_scale, op_zero_point)

# torch.onnx.register_custom_op_symbolic(
#         'quantized::conv_transpose2d', convert_quant_conv_transpose2d, 13)

import torch
import random

x = torch.ones(2, 6, 51, 52)#(5, 3)
scale = random.uniform(0., 2.)
zp = random.randint(0, 5)
input_shapes = [x.shape]

spatial_size = len(input_shapes[0]) - 2

in_channels= input_shapes[0][1]
out_channels =6

stride = [1,1]

padding  = [0,0]
dilation =[1,1]
groups = 1
bias = True
output_padding = [0,0]

kernel_size=3
kernel_size = [kernel_size] * spatial_size

weight_shape = [input_shapes[0][1], out_channels // groups] + list(kernel_size)

w = torch.quantize_per_tensor(torch.randn(weight_shape), scale=0.5, zero_point=2, dtype=torch.qint8)


class test_module(torch.nn.Module):
    def __init__(self):
        super(test_module, self).__init__()
        self.deconv = torch.nn.quantized.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        if bias:
            b = torch.randn(out_channels)
        else:
            b = torch.zeros(out_channels)
        self.deconv.set_weight_bias(w, b)
        self.deconv.scale = scale
        self.deconv.zero_point = zp

    def forward(self, x):
        x = torch.quantize_per_tensor(x, scale=2.1, zero_point=2, dtype=torch.quint8)
        return self.deconv(x)


model1 = test_module()
input=x
out1=model1(input)

print('..')
# traced_model= torch.jit.trace(model,input_batch,strict=False)
traced_model= torch.jit.trace(model1,input)
traced_model.save('torch_quant_convtranspose.pt')

# script_model=torch.jit.script(model)
# script_model.save('deep_lab_v3_script.pt')


# For inference
model1.eval()

from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args,quantized_args,dequantize_helper,_unpack_list
from torch.onnx import symbolic_opset9 as opset9

# @parse_args("v", "v", "f", "f")
# def convert_quant_conv_transpose2d(g, q_input, q_weight, output_scale, output_zero_point):
#     inp, input_scale, _, _ = dequantize_helper(g, q_input)
#     unpacked_inputs = _unpack_param(q_weight)#2 6 51 52
#     #output = opset9.conv2d(g, input, weight, bias, stride, padding, dilation, groups)
#     #return quantize_helper(g, output, op_scale, op_zero_point)
#
#
# torch.onnx.register_custom_op_symbolic(
#         'quantized::conv_transpose2d', convert_quant_conv_transpose2d, 13)

# optionally, if you want to export the model to ONNX:
torch.onnx.export(traced_model, input, "torch_conv_transpose.onnx", opset_version = 13)

[tasklist]
### Tasks