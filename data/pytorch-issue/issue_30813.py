import torch

t = torch.rand(10)
print(t)
# tensor([0.6088, 0.3496, 0.3973, 0.0884, 0.5340, 0.9819, 0.5057, 0.2072, 0.6677,
#         0.6197], device='cuda:0')
t = torch.quantize_per_tensor(t, 0.01, 0, torch.qint8)
print(t)
# tensor([0.6100, 0.3500, 0.4000, 0.0900, 0.5300, 0.9800, 0.5100, 0.2100, 0.6700,
#         0.6200], size=(10,), dtype=torch.qint8, device='cuda:0',
#        quantization_scheme=torch.per_tensor_affine, scale=0.01, zero_point=0)
t = t.dequantize()
print(t)
# tensor([0.6100, 0.3500, 0.4000, 0.0900, 0.5300, 0.9800, 0.5100, 0.2100, 0.6700,
#         0.6200], device='cuda:0')