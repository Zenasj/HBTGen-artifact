import random

import numpy as np
import torch
n_gpus = torch.cuda.device_count()
streams = [torch.cuda.Stream(device=i) for i in range(n_gpus)]

x = torch.tensor(np.random.rand(100, 2048, 2048).astype(np.float32))
xs = [x.to(i) for i in range(n_gpus)]
Xgs = []

# Start timing here
for i, s in enumerate(streams):
    with torch.cuda.stream(s):
        for _ in range(100):
            Xg = torch.rfft(xs[i], 2)
    Xgs.append(Xg)

# Indent below to run explicitly in serial
torch.cuda.synchronize()

for _ in range(1000):
    for i, s in enumerate(streams):
        Xg = torch.rfft(xs[i], 2)