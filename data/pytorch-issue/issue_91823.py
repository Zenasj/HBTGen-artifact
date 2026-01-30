import torch
import tracemalloc

tracemalloc.start()
x = torch.ones((3, 1024, 1024, 1024), dtype=torch.uint8)
current, peak = tracemalloc.get_traced_memory()
print(f'Current: {current / 1024**2:.2f} [MB], peak: {peak / 1024**2:.2f} [MB]')

import tracemalloc
import numpy as np

tracemalloc.start()
a = np.ones((3, 1024, 1024, 1024), dtype=np.uint8)
current, peak = tracemalloc.get_traced_memory()
print(f'Current: {current / 1024**2:.2f} [MB], peak: {peak / 1024**2:.2f} [MB] / 1024**2')