# heavy.py
import torch

x = torch.randn(4*1024, 4*1024, device="cuda")
print("start")

for _ in range(1000):
    torch.matmul(x, x)

torch.cuda.synchronize()
print("done")

# light.py
import torch
import time

x = torch.randn(4*1024, 4*1024, device="cuda")
print("start")

for _ in range(100):
    torch.matmul(x, x)
    time.sleep(0.1)

torch.cuda.synchronize()
print("done")