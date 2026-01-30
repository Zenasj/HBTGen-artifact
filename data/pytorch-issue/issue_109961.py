import torch
from torch.nn import Linear

try:
    model = Linear(64, 64)
except torch.cuda.OutOfMemoryError:
    print("Out of memory")