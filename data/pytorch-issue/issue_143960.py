import torch
import numpy as np
torch.manual_seed(5)

dist = torch.dist
compiled_dist = torch.compile(torch.dist)
incon1 = []
incon2 = []
for i in range(100):
    a = torch.rand(1).float()
    b = torch.rand(1).float()
    high_a = a.double()
    high_b = b.double()
    ref = compiled_dist(high_a, high_b, -41)
    incon1.append(torch.abs(dist(a, b, -41) - ref))
    incon2.append(torch.abs(compiled_dist(a, b, -41) - ref))
print("Average error before compile: ", np.average(incon1))
print("Average error after compile: ", np.average(incon2))