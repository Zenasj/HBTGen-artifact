import os
import numpy as np
import psutil
import torch
from tqdm import trange

if __name__ == "__main__":
    print("\n\n\n")
    n_samples = 100000

    pid = os.getpid()
    prev_memoryUse = 0.0

    py = psutil.Process(pid)
    init_mem = py.memory_info()[0] / 2. ** 30
    print(f"PyTorch Version {torch.__version__}")
    for i in trange(n_samples, desc="Torch Memory:", ncols=100):
        tmp = torch.matmul(torch.zeros((1, 256, 256)), torch.zeros((1, 256, 2)))

    memoryUse = py.memory_info()[0] / 2. ** 30
    print(f"Torch Memory: Memory = {memoryUse:.3e}Gb \t Delta Memory = {memoryUse - init_mem:+.3e}Gb")

OMP_NUM_THREADS=4

import torch
torch.eye(252)

import torch
import tqdm

a = torch.zeros((8, 256, 256))
for _ in tqdm.trange(10000):
    a[0].fill_(1)
    a.fill_(0)