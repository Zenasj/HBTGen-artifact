import torch

torch.backends.quantized.engine = 'qnnpack'
print(torch.backends.quantized.engine, torch.quantize_per_tensor(torch.randn(5, 5, 5, 5), scale=0.2, zero_point=0, dtype=torch.quint8).mean((2,3), keepdim=True).shape)
torch.backends.quantized.engine = 'fbgemm'
print(torch.backends.quantized.engine, torch.quantize_per_tensor(torch.randn(5, 5, 5, 5), scale=0.2, zero_point=0, dtype=torch.quint8).mean((2,3), keepdim=True).shape)