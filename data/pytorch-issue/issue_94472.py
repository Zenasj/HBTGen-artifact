py
import torch
import time

while True:
    print(torch.cuda.utilization())
    time.sleep(1)

py
import torch

while True:
    t = torch.ones(10000, 10000).cuda()
    t = t * t