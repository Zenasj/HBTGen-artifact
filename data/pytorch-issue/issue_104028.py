import torch
import time

tsize = 100
torch.manual_seed(37)
torch.set_num_threads(1)
repeats = 10_000_000
print("Torch version:",torch.__version__)
a = torch.randn(tsize, dtype=torch.float)
b = torch.arange(0, tsize - 1, dtype=torch.long)
values = torch.randn(b.shape, dtype=torch.float)

# warm up
for i in range(10_000):
    a[b] = values

begin = time.time()
for i in range(repeats):
    a[b] = values

end = time.time()

print("Time : ",  (end - begin) * 1000.0 / repeats)

import torch
import time

tsize = 1000
torch.manual_seed(37)
torch.set_num_threads(1)
repeats = 10_000_000
print("Torch version:",torch.__version__)
a = torch.randn(tsize, dtype=torch.float)
b = torch.arange(0, tsize - 1, dtype=torch.long)

# warm up
for i in range(10_000):
    result = a.index_select(0,b)

begin = time.time()
for i in range(repeats):
    result = a.index_select(0,b)

end = time.time()

print("Time : ",  (end - begin) * 1000.0 / repeats)