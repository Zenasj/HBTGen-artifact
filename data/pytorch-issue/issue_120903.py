import torch
import numpy as np

input_data = torch.randn(3, 4, 5, 6)

scale = np.array([0.1, 0.2, 0.3, 0.4]) 
zero_point = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
axis = 1
quant_min = 0
quant_max = 255

output = torch.fake_quantize_per_channel_affine(input_data, torch.from_numpy(scale), zero_point, axis, quant_min, quant_max)

print(output)