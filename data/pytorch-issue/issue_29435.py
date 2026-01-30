import torch
x = torch.randn(62, 59)
qx = torch.quantize_per_tensor(x, 1.0, 0, torch.qint32)
qy = qx.permute([1, 0])
qy.dequantize()

import torch
x = torch.randn(64, 64)
qx = torch.quantize_per_tensor(x, 1.0, 0, torch.qint32)
print(qx.permute([1, 0]))