import time
import torch

DELAY = 100000000 
x = torch.randn((1024, 1024), pin_memory=True)

torch.cuda.synchronize()
start = time.time()
torch.cuda._sleep(DELAY)
x.cuda(non_blocking=True)
end = time.time()

print('non_blocking=True', (end - start)*1000.)  # ~7 ms on my GPU

torch.cuda.synchronize()
start = time.time()
torch.cuda._sleep(DELAY)
x.cuda(non_blocking=False)
end = time.time()


print('non_blocking=False', (end - start)*1000.)  # ~77 ms on my GPU