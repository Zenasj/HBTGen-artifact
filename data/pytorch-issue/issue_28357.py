import torch
import torch.nn as nn

scale, zero_point, dtype = 1.0, 2, torch.qint8
quantize_tensor = torch.nn.quantized.Quantize(scale, zero_point, dtype)
input_tensor=quantize_tensor(input_tensor)