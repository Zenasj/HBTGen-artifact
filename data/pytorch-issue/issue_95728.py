import itertools
import torch
import time
t0 = time.time()
world_size = 1024
ws = torch.range(0, world_size)
ws = ws.cuda() # or cpu() to compare
c = itertools.combinations( ((rank,ws[rank]) for rank in range(world_size)), 2)
for (r1, n1), (r2, n2) in c:
    if n1 != n2:
        pass
print(time.time() - t0)