import torch
import time

a = torch.randn(1000, 1000).cuda()
b = torch.randn(1000, 1000).cuda()

ops = torch.mm

# warm up
for i in range(100):
    ops(a,b)

# record time
start_time = time.time()
for i in range(100000):
    ops(a,b)
end_time = time.time()

elapsed_time = end_time - start_time

print("torch.mm cost time is ：", elapsed_time, "sec")