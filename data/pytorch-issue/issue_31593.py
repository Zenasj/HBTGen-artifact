import torch
import torch.nn as nn
import time

nb_iters = 10
sizes = [int(50000/2**i) for i in range(10)]

for size in sizes:
    x = torch.randn(size, 1, device='cuda', requires_grad=True)
    # warmup
    for _ in range(nb_iters):
        out = torch.pdist(x)
        out.mean().backward()
    #print(torch.cuda.memory_allocated()/1024**3)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(nb_iters):
        out = torch.pdist(x)
        out.mean().backward()
    torch.cuda.synchronize()
    t1 = time.time()
    print('size {}, time {:.4f}ms/iter'.format(size, 1000*(t1 - t0)/nb_iters))