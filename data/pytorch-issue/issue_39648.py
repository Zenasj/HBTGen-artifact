import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist

def main():
    SEED = 1375 # random seed for reproduce results
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    ngpus_per_node = torch.cuda.device_count()
    world_size = 1 * ngpus_per_node
    mp.spawn(main_worker, nprocs=ngpus_per_node)

def main_worker(gpu):
    ...