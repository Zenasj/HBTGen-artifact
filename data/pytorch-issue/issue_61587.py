import torch.nn as nn

import torch
device = "cuda"

# Time pytorch BN1d
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

x = torch.randn(100000,32).to(device)
bn = torch.nn.BatchNorm1d(32).to(device)

N=100
start.record()
for i in range(N):
    out = bn(x)
end.record()

torch.cuda.synchronize()
print("BN1d Forward:", start.elapsed_time(end)/N)

# Time Naive implementation
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

x = torch.randn(100000,32).to(device)
bn = MyBN1(32).to(device)

N=100
start.record()
for i in range(N):
    out = bn(x)
end.record()

torch.cuda.synchronize()
print("Naive BN1d Forward:",start.elapsed_time(end)/N)