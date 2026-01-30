import torch
import torch.nn as nn
from torch.multiprocessing import Pool, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

class Dummy:
    def __init__(self, device):
        self.device = device
        self.layer = nn.Linear(100, 100).to(self.device).share_memory() # removing share_memory doesn't have any effects
    def run(self):
        self.layer(torch.rand(1000, 100).to(self.device))

def run_steps(agent):
    for k in range(100):
        agent.run()

n_gpus = 2
agents = [Dummy("cuda:%d" % k) for k in range(n_gpus)]

# THIS HANGS
pool = Pool(n_gpus)
pool.map(run_steps, agents)

# This works:
from multiprocessing.dummy import Pool as dThreadPool
pool = dThreadPool(n_gpus)
pool.map(run_steps, agents)

import torch
import torch.nn as nn
from time import time
from torch.multiprocessing import Pool, set_start_method, freeze_support
from multiprocessing.dummy import Pool as dThreadPool
try:
     set_start_method('spawn')
except RuntimeError:
    pass

class Dummy:
    def __init__(self, device):
        self.device = device
        self.layer = nn.Linear(100, 100).to(self.device).share_memory() # removing share_memory doesn't have any effects
    def run(self):
        p = torch.rand(1000, 100).to(self.device)
        p = self.layer(p)

def run_steps(agent):
    for k in range(n_iters):
        agent.run()

n_iters = 100

if __name__ == '__main__':
    freeze_support()
    n_gpus = 2
    agents = [Dummy("cuda:%d" % k) for k in range(n_gpus)]

    start = time()
    pool = Pool(n_gpus)
    pool.map(run_steps, agents)
    print("It took %g" % (time()-start))
    pool.close()