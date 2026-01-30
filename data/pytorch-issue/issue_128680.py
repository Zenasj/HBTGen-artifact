import torch.nn as nn

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
class TwoLinLayerNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(10, 10, bias=False)
        self.b = torch.nn.Linear(10, 1, bias=False)
 
    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        return (a, b)
 
def worker(rank):
    dist.init_process_group("nccl", rank=rank, world_size=int(os.environ.get('WORLD_SIZE')))
    torch.cuda.set_device(rank)
    print("init model")
    model = TwoLinLayerNet().cuda()
    print("init ddp")
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
 
    inp = torch.randn(10, 10).cuda()
    print("train")
 
    for _ in range(20):
        output = ddp_model(inp)
        loss = output[0] + output[1]
        loss.sum().backward()
 
if __name__ == "__main__":
    mp.spawn(worker, nprocs=int(os.environ.get('WORLD_SIZE')), args=())