py
import torch
import torch.nn as nn
import time
import os

torch.manual_seed(0)

def _time():
    return time.time()

device = torch.device(os.getenv('DEVICE'))
m = nn.LogSigmoid()
#warm up
for n in [1, 10, 100, 1000]:
    input = torch.randn(128, n, requires_grad=True, device=device)
    grad_output = torch.randn(128, n, device=device)
    for i in range(1000):
        output = m(input)
        output.backward(grad_output)

for n in [1, 10, 100, 1000]:
    input = torch.randn(128, n, requires_grad=True, device=device)
    grad_output = torch.randn(128, n, device=device)
    fwd_t = 0
    bwd_t = 0
    for i in range(10000):
        t1 = _time()
        output = m(input)
        t2 = _time()
        output.backward(grad_output)
        t3 = _time()
        fwd_t = fwd_t + (t2 - t1)
        bwd_t = bwd_t + (t3 - t2)
    fwd_avg = fwd_t / 10000 * 1000
    bwd_avg = bwd_t / 10000 * 1000
    print("input size(128, %d) forward time is %.2f (ms); backwad avg time is %.2f (ms)."
          % (n, fwd_avg, bwd_avg))