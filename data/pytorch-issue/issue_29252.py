import numpy as np
import torch

x = np.ones(1, dtype='int64')[0]
y = torch.ones(size=(1,), dtype=torch.int64)
z = x+y
print(f"z.dtype: {z.dtype}")

z.dtype: torch.float32