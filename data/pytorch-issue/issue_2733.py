import torch.distributed as dist
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=0, type=int)
args = parser.parse_args()
dist.init_process_group(backend="mpi", init_method="env://",
                       world_size = int(os.environ["WORLD_SIZE"]), rank=args.local_rank)
## Model.
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))

model = Model() 
net = torch.nn.parallel.DistributedDataParallelCPU(model)