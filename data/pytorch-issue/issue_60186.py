import torch
import math
import numpy as np

# Single Precision
print(torch.tan(torch.tensor(math.pi, device='cuda', dtype=torch.float32) * 0.5))
print(np.tan(np.array(np.pi, dtype=np.float32) * 0.5))

# Double Precision
print(math.tan(math.pi * 0.5))
print(torch.tan(torch.tensor(math.pi, device='cuda', dtype=torch.double) * 0.5))
print(np.tan(np.array(np.pi, dtype=np.float64) * 0.5))

import torch
import itertools
import time
from torch.utils.benchmark import Timer
from torch.utils.benchmark import Compare
import sys
import pickle

print('Using pytorch %s' % (torch.__version__))

cuda_shapes = [(128,), (256, 128), (2, 512, 256), (2, 64, 256, 128), (4, 2, 512, 256, 128)]
cuda_dtypes = [torch.half, torch.bfloat16, torch.float, torch.double]
results = []
repeats = 10

for device in ['cuda']:
    dtypes = cuda_dtypes
    shapes = cuda_shapes

    for dtype in dtypes:
        for shape in shapes:
            t = torch.randn(shape, device=device, dtype=dtype) * 10
            
            tasks = [("t.cauchy_()", "After", "")]
            timers = [Timer(stmt=stmt, label=f"cauchy : input dtype {dtype} device {device}", sub_label=f"{(shape)}", description=desc, globals=globals()) for stmt, desc, label in tasks]

            for i, timer in enumerate(timers * repeats):
                results.append(
                    timer.blocked_autorange()
                )
                print(f"\r{i + 1} / {len(timers) * repeats}", end="")
                sys.stdout.flush()

with open('after-pr.pkl', 'wb') as f:
    pickle.dump(results, f)

comparison = Compare(results)
comparison.print()