import torch

t = torch.arange(4.0)
q = torch.quantize_per_tensor(t, 0.02, 0, torch.qint8)
q2 = torch._empty_affine_quantized(q.shape, scale=0.04, zero_point=0, dtype=torch.qint8)
q3 = torch._empty_affine_quantized(q.shape, scale=0.04, zero_point=0, dtype=torch.qint8)
q2[:] = q
q3[:] = q.dequantize()
q, q2, q3