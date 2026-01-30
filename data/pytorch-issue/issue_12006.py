import torch.nn as nn

# test_bn.py
import torch
import time
import sys

bs = int(sys.argv[1])
num_features = 256

shape = torch.Size((bs * 20000, num_features))
input = torch.cuda.FloatTensor(shape)
torch.randn(shape, out=input)
input.requires_grad = True

bn = torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
bn.cuda()

N = 30
forward_time = 0
backward_time = 0
for i in range(N):
    torch.cuda.synchronize()
    start = time.time()
    output = bn(input)
    output = output.mean()
    torch.cuda.synchronize()
    forward_time += time.time() - start

    torch.cuda.synchronize()
    start = time.time()
    output.backward()
    torch.cuda.synchronize()
    backward_time += time.time() - start

print('forward: %f', forward_time / N)
print('backward: %f', backward_time / N)
print('total: %f', (forward_time + backward_time) / N)