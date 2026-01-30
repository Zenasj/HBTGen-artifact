import torch.nn as nn

import torch
import numpy as np

torch.manual_seed(1)
input_tensor = torch.rand(1, 28, 514, 514, 256)

conv = torch.nn.Conv3d(28, 12, (1, 1, 1))
conv.reset_parameters()

conv_result = conv(input_tensor)
print(f"Conv result with large input: {conv_result[0, 0, 0, 0, 0]}")
# Conv result with large input: -0.06907586008310318

with torch.backends.mkldnn.flags(enabled=False):
    conv_result = conv(input_tensor)
print(f"Conv result with large input and no mkldnn: {conv_result[0, 0, 0, 0, 0]}")
# Conv result with large input and no mkldnn: -0.05493345856666565

conv_result = conv(input_tensor[:, :, :256, :256, :])
print(f"Conv result with smaller input: {conv_result[0, 0, 0, 0, 0]}")
# Conv result with smaller input: -0.05493343621492386

numpy_input = input_tensor.detach().numpy()
weight_for_first_channel = np.squeeze(conv.weight[0].detach().numpy())
bias_for_first_channel = conv.bias[0].detach().numpy()
numpy_result = np.sum(numpy_input[0, :, 0, 0, 0]* weight_for_first_channel) + bias_for_first_channel
print(f"Numpy result: {numpy_result}")
# Numpy result: -0.054933398962020874