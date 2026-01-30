import torch
import numpy as np

qcat = torch.ops.quantized.cat

s = 1.479488579387789e+37
z = 23


x = np.array([[[[0.]]]], dtype=np.float32)
x = np.repeat(x, 70/x.shape[3], 3)

x = torch.from_numpy(np.ascontiguousarray(x))
y = x.clone()

qx = torch.quantize_per_tensor(x, s, z, torch.quint8).permute([0, 3, 1, 2])
qy = torch.quantize_per_tensor(y, s, z, torch.quint8).permute([0, 3, 1, 2])

print(qcat([qx, qy], scale=s, zero_point=z, dim=1))