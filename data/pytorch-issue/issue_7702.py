# test.py
import time
import random

import torch
import torch.distributed as dist
def print_mem(extra_str=''):
    if dist.get_rank() == 0:
        import os
        import psutil
        process = psutil.Process(os.getpid())
        print(extra_str, process.memory_info().rss // 2**20, 'MB')

dist.init_process_group('gloo', world_size=2, init_method='file:///tmp/shared_file')
embedding = torch.randn(8000, 200)
for epo in range(1000):
    index = torch.randint(embedding.size(0), (30000 - epo,)).long()
    # ten = torch.randn(numel)
    ten = embedding[index]
    print_mem('before reducing')
    dist.all_reduce(ten, op=dist.reduce_op.SUM)
    print_mem('after reducing')