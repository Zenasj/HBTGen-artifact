import torch.nn as nn

import time
import torch


weight_norm = torch.nn.utils.parametrizations.weight_norm
conv_layer = torch.nn.Conv1d(in_channels=192, out_channels=383, kernel_size=5, dilation=1, padding=2, dtype=torch.bfloat16)
in_layer = weight_norm(conv_layer)
input_tensor = torch.rand(1, 192, 178).to(conv_layer.weight.dtype) - 0.5
with torch.no_grad():
    for i in range(100):
        start = time.time()
        out = in_layer(input_tensor)
        end = time.time()
        print(f"time costs: {(end-start)*1000000} us")