import torch
import numpy as np
np.clip(np.array([1], dtype=np.float32), np.array([1], dtype=np.int32), None).dtype  # dtype('float64')
torch.clamp(torch.tensor([1], dtype=torch.float32), torch.tensor([1], dtype=torch.int32)).dtype  # torch.float32