import torch.nn as nn

import torch
import torch.nn.functional as F

import time
filters = torch.ones(1, 1, 67, 67)
inputs = torch.randn(1, 1, 1410, 1280)
start_time_total = time.time()
for i in range(100):
    start_time = time.time()
    res = F.conv2d(inputs, filters, padding=1)
    print(f"conv2d: {time.time() - start_time} s")
print(f"----------conv2d: {(time.time() - start_time_total)/100} s")