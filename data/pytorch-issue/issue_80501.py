import torch
torch.manual_seed(20)

w = torch.randn(39979771, 128)
scales =  0.00187
zp = 0

i8_arg = torch.quantize_per_tensor(w, scales, zp, torch.qint8)
arg = i8_arg.dequantize()
print(w[10622505])
print(i8_arg[10622505])