import torch
import torch.nn as nn
import torch.nn.functional as F

qF.conv2d(q_inputs, q_filters, bias, scale=scale, zero_point=zero_point, padding=1)

qF.conv2d(q_inputs, q_filters, bias)

from torch.nn.quantized import functional as qF
filters = torch.randn(8, 4, 3, 3, dtype=torch.float)
inputs = torch.randn(1, 4, 5, 5, dtype=torch.float)
bias = torch.randn(8, dtype=torch.float)

scale, zero_point = 1.0, 0
dtype = torch.quint8

# torch.backends.quantized.engine = 'qnnpack'

q_filters = torch.quantize_per_tensor(filters, scale, zero_point, dtype)
q_inputs = torch.quantize_per_tensor(inputs, scale, zero_point, dtype)
# qF.conv2d(q_inputs, q_filters, bias, scale=scale, zero_point=zero_point, padding=1)
qF.conv2d(q_inputs, q_filters, bias, scale, zero_point, padding=1)