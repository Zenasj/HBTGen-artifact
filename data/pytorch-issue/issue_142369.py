import torch
from torch._inductor.cpu_vec_isa import pick_vec_isa
import time

start = time.time()
out = pick_vec_isa()
end = time.time()
# 9.106s on my machine
print("pick isa first time(no cache):", end - start)

start = time.time()
out = pick_vec_isa()
end = time.time()
# 9.106s on my machine
print("pick isa second time(with cache):", end - start)