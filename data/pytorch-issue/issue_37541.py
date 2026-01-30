import torch
import time

print(torch.__version__)

a = [torch.rand(256, 80, 80) for _ in range(20)]

t = time.perf_counter()
for _ in range(1000):
    for i in range(len(a)):
        torch.relu(a[i])
print(time.perf_counter() - t, "loop")

t = time.perf_counter()
for _ in range(1000):
    for i in range(len(a)):
        a[i].relu_()
print(time.perf_counter() - t, "loop inplace")

a = torch.stack(a)

t = time.perf_counter()
for _ in range(1000):
    torch.relu(a)
print(time.perf_counter() - t, "noloop")

t = time.perf_counter()
for _ in range(1000):
    a.relu_()
print(time.perf_counter() - t, "noloop inplace")

a = [torch.rand(256, 80, 80, device="cuda") for _ in range(20)]

t = time.perf_counter()
for _ in range(1000):
    for i in range(len(a)):
        torch.relu(a[i])
print(time.perf_counter() - t, "cuda loop")

t = time.perf_counter()
for _ in range(1000):
    for i in range(len(a)):
        a[i].relu_()
print(time.perf_counter() - t, "cuda loop inplace")

a = torch.stack(a)

t = time.perf_counter()
for _ in range(1000):
    torch.relu(a)
print(time.perf_counter() - t, "cuda noloop")

t = time.perf_counter()
for _ in range(1000):
    a.relu_()
print(time.perf_counter() - t, "cuda noloop inplace")