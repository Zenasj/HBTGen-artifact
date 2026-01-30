import torch
num_elements=10
r = torch.ones(num_elements, dtype=torch.float, device='cpu')
scale = 1.0
zero_point = 2
qr = torch.quantize_per_tensor(r, scale, zero_point, torch.qint8)
x = torch.randn(10)

# x is strided CPUTensor, qr is QuantizedCPU tensor
x[qr]