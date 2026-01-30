import torch.nn as nn

python
""" test.py """
import os
import sys
import gc
import logging
import time
import torch
import torch.distributed as dist
from torch.distributed import get_rank, get_world_size

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print (rank, size)
os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '23333'
os.environ['WORLD_SIZE'] = str(size)
os.environ['RANK'] = str(rank) 
dist.init_process_group(backend="gloo")

print ("initialize gloo successfully [rank {}] pid({})".format(rank, os.getpid()))

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    batch_size = 16
    feat_dim = 2048
    pid = os.getpid()
    prev_mem = 0
    tensor_list = [torch.FloatTensor(batch_size, feat_dim) for _ in range(get_world_size())]
    values = torch.FloatTensor(batch_size, feat_dim)
    for idx in range(10000):
        if idx % 20 == 0 and get_rank() == 0:
            cur_mem = (int(open('/proc/%s/statm' % pid, 'r').read().split()[1]) + 0.0) / 256
            add_mem = cur_mem - prev_mem
            prev_mem = cur_mem
            print ("train iterations: %s, added mem: %s M" % (idx, add_mem))
        dist.all_gather(tensor_list=tensor_list, tensor=values)