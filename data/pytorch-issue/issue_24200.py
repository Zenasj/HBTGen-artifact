import numpy as np
import torch

def mem():
	import os, psutil
	return psutil.Process(os.getpid()).memory_info().rss // 1024

a = np.zeros(10000, dtype = np.float32) # <-- 1. dtype is important
a = list(a) # <-- 2. has to be a python list

mem_base = mem()
for _ in range(10):
	torch.tensor(a) # <-- 3. must not pass `dtype=torch.float32` here
	print(mem() - mem_base)

# output on my machine:
# 1416
# 2884
# 3940
# 4996
# 6316
# 7372
# 8428
# 9484
# 10804
# 11860

class Dataset:
    def __init__(self, X, y):
        self.X = x; self.y = y
    
    def __getitem__(self, i):
        return self.X[i], self.y[i] # <-- self.y[i] is a np.float32
    
    def __len__(self):
        return self.X.shape[0]