import numpy as np
import torch

X = torch.from_numpy(np.full(64+1, 514., dtype=np.float32))
(scale, zero_point, torch_type) = (1028.02, 255, torch.quint8)

assert X.is_contiguous(memory_format=torch.contiguous_format)
qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                               dtype=torch_type)

f_min, f_max = 0.0, 1.0
q_min, q_max = torch.iinfo(torch_type).min, torch.iinfo(torch_type).max
output_scale = (f_max - f_min) / (q_max - q_min + 1.0)

qY = torch.ops.quantized.sigmoid(qX, output_scale=output_scale, output_zero_point=0)
print(qY)
assert qY[0] == qY[-1]