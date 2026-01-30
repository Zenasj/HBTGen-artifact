import torch

import time
def compute(X, Y, W):
    # time.sleep(1)
    E = W > 0
    T, F = (Y >= 0.5) & E, (Y < 0.5) & E
    P, N = (X >= 0) & E, (X < 0) & E
    return sum(T&P).numpy(), sum(F&P).numpy(), sum(T&N).numpy(), sum(F&N).numpy()
  
X = torch.randn(100, 100, 100).cuda()
Y = torch.rand(100, 100, 100).cuda()
Z = torch.rand(100, 100, 100).cuda()
Z[Z<0.1] = 0
Y = X.cuda()
A = compute(*map(lambda x: x.flatten(0, 1).cpu(), [X, Y, Z]))
B = compute(*map(lambda x: x.flatten(0, 1).to("cpu", non_blocking=True), [X, Y, Z]))

for a, b in zip(A, B):
    print(a == b)