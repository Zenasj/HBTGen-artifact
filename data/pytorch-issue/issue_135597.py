import torch

quant_scale_tensor = getattr("quant_scale")
quant_scale_float = float(quant_scale_tensor)
quantized = quantize_per_tensor(x, quant_scale_float, ...)

quant_scale_tensor = getattr("quant_scale")
quant_scale_symfloat = quant_scale_tensor.to(dtype=torch.float).item()
quantized = quantize_per_tensor(x, quant_scale_symfloat, ...)