import torch
import math
import numpy as np

n = 1e9

torch.remainder(torch.tensor([n], device="cpu", dtype=torch.float32), math.pi)  # will return tensor([-64.]) which is larger than the divisor math.pi

torch.remainder(torch.tensor([n], device="cuda", dtype=torch.float32), math.pi)  # will return tensor([-33.5333], device='cuda:0') which is also larger than the divisor math.pi

np.remainder(np.array([n], dtype=np.float32), math.pi)  # will return array([1.024195], dtype=float32)