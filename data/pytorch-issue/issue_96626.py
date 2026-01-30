3
import time
import torch

def test(x):
    torch.cuda.synchronize()
    t1 = time.time()
    torch.bincount(x)
    torch.cuda.synchronize()
    t2 = time.time()
    return t2 - t1

print("case 1 CPU ", test(torch.ones(2**18, dtype=torch.int32) * 2048), "seconds")
print("case 1 CUDA", test(torch.ones(2**18, dtype=torch.int32).cuda() * 2048), "seconds")
print("case 2 CPU ", test(torch.ones(2**20, dtype=torch.int32) * 2048), "seconds")
print("case 2 CUDA", test(torch.ones(2**20, dtype=torch.int32).cuda() * 2048), "seconds")