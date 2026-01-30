import torch.nn as nn

import torch

# define a floating point model where some layers could be statically quantized
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.relu = torch.nn.ReLU()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.conv(x)
        x = self.relu(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

# create a model instance
model_fp32 = M()

# model must be set to eval mode for static quantization logic to work
model_fp32.eval()

# attach a global qconfig, which contains information about what kind
# of observers to attach. Use 'fbgemm' for server inference and
# 'qnnpack' for mobile inference. Other quantization configurations such
# as selecting symmetric or assymetric quantization and MinMax or L2Norm
# calibration techniques can be specified here.
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Fuse the activations to preceding layers, where applicable.
# This needs to be done manually depending on the model architecture.
# Common fusions include `conv + relu` and `conv + batchnorm + relu`
model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'relu']])

# Prepare the model for static quantization. This inserts observers in
# the model that will observe activation tensors during calibration.
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

# calibrate the prepared model to determine quantization parameters for activations
# in a real world setting, the calibration would be done with a representative dataset
input_fp32 = torch.randn(4, 1, 4, 4)
model_fp32_prepared(input_fp32)

# Convert the observed model to a quantized model. This does several things:
# quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, and replaces key operators with quantized
# implementations.
model_int8 = torch.quantization.convert(model_fp32_prepared)

model_int8(input_fp32)
print(model_int8)
"""
M(
  (quant): Quantize(scale=tensor([0.0279]), zero_point=tensor([56]), dtype=torch.quint8)
  (conv): QuantizedConvReLU2d(1, 1, kernel_size=(1, 1), stride=(1, 1), scale=0.007665436714887619, zero_point=0)
  (relu): Identity()
  (dequant): DeQuantize()
)
"""

import copy
model_copied = copy.deepcopy(model_int8)
model_copied(input_fp32)
"""
Traceback (most recent call last):
  File "q_example.py", line 71, in <module>
    model_copied(input_fp32)
  File "/data/users/jamesreed/pytorch/torch/nn/modules/module.py", line 1015, in _call_impl
    return forward_call(*input, **kwargs)
  File "q_example.py", line 18, in forward
    x = self.conv(x)
  File "/data/users/jamesreed/pytorch/torch/nn/modules/module.py", line 1013, in _call_impl
    if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
  File "/data/users/jamesreed/pytorch/torch/nn/modules/module.py", line 1095, in __getattr__
    type(self).__name__, name))
AttributeError: 'ConvReLU2d' object has no attribute '_backward_hooks'
"""
print(model_copied)
"""
Traceback (most recent call last):
  File "q_example.py", line 62, in <module>
    print(model_copied)
  File "/data/users/jamesreed/pytorch/torch/nn/modules/module.py", line 1686, in __repr__
    mod_str = repr(module)
  File "/data/users/jamesreed/pytorch/torch/nn/modules/module.py", line 1685, in __repr__
    for key, module in self._modules.items():
  File "/data/users/jamesreed/pytorch/torch/nn/modules/module.py", line 1095, in __getattr__
    type(self).__name__, name))
AttributeError: 'ConvReLU2d' object has no attribute '_modules'
"""

import torch

class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)

f = Foo()
f(torch.randn(5, 3))

import copy
copied = copy.deepcopy(f)
copied(torch.randn(5, 3))