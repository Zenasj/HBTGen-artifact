import torch 
import multiprocessing as mp
try:
    mp.set_start_method('spawn') # spawn, forkserver, and fork
except RuntimeError:
    pass

torch.distributed.init_process_group(backend='nccl', world_size=2, init_method='tcp://224.66.41.62:23456')