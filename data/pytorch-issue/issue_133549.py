# torchrun --nproc_per_node=4 test_dtensor.py

import os
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh

mesh = init_device_mesh("cuda", (int(os.environ["WORLD_SIZE"]),))

def test_raw_tensor():
    big_tensor = torch.randn((4, 1024*1024*1024)).to("cuda")

def test_shard():
    big_tensor = torch.randn((4, 1024*1024*1024))
    my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=0)])

def test_replicate():
    big_tensor = torch.randn((4, 1024*1024*1024))
    my_dtensor = distribute_tensor(big_tensor, mesh, [Replicate()])

# test_raw_tensor()
test_shard()
# test_replicate()

print("wait for gpu inspect...")
import time
time.sleep(60)