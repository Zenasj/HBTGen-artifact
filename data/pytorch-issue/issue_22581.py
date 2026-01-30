import torch

x = torch.randn(1, 4, 4, 4)
x = x.transpose(0, 1)
for i in range(10):
    # note: results are often different on each run
    # or on CPU, outputs `nan`
    print(x.triu().sum())