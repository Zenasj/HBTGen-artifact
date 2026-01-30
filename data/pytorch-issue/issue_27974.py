3
import time
import numpy as np
import torch
import torch.nn as nn

n, c, h, w = (32, 8, 32, 32)
device = torch.device("cpu")
model = nn.Conv2d(c, c, 5, padding=2)
model = model.to(device)
model.train()
loss_fn = nn.MSELoss()

for iteration in range(100):
    t0 = time.perf_counter()
    
    x = torch.from_numpy(np.ones((n, c, h, w), np.float32)).to(device)
    y = torch.from_numpy(np.ones((n, c, h, w), np.float32)).to(device)
    loss = loss_fn(model(x), y)
    
    t1 = time.perf_counter()
    
    loss.backward()
    
    t2 = time.perf_counter()
    print("forward: %f seconds" % (t1 - t0))
    print("backward: %f seconds" % (t2 - t1))
    print("")