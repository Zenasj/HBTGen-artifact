import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear
from torch.optim import SGD
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo.backends.distributed import DDPOptimizer
import torch._dynamo as dynamo
from torch._inductor.compile_fx import compile_fx

os.environ["TORCH_COMPILE_DEBUG"] = "0"
dynamo.reset()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="GLOO", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.lin1 = Linear(3 * 1024, 2 * 1024, bias=False)
        self.lin2 = Linear(2 * 1024, 2 * 1024, bias=False)
        self.lin3 = Linear(2 * 1024, 3 * 1024, bias=False)

    def forward(self, x):
        t1 = self.lin1(x).relu().add(1.0)
        t2 = self.lin2(t1).relu().pow(2.0)
        c1 = torch.cat((t1, t2))
        t3 = self.lin3(c1).relu().mul(2.0)
        return t3

def simple(rank, world_size, iterations):
    setup(rank, world_size)
    model = ToyModel()
    model = DDP(model, bucket_cap_mb=70)

    intensor = torch.rand(8 * 1024, 3 * 1024)
    ddp_optimizer = DDPOptimizer(
              bucket_bytes_cap=int(100 * 1024 * 1024),
              backend_compile_fn=compile_fx
          )

    @torch._dynamo.optimize(ddp_optimizer.compile_fn, dynamic=True)
    def opt_fn(inp):
          return model(inp)

    optim = SGD(model.parameters(), lr=0.1)

    for i in range(iterations):
          out = opt_fn(intensor)
          out.float().sum().backward()
          optim.step()
          optim.zero_grad()

    print(f"Done for rank :: {rank}")
    cleanup()

if __name__ == "__main__":
    WORLD_SIZE = 2
    mp.start_processes(simple,
        args=(WORLD_SIZE, 5),
        nprocs=WORLD_SIZE,
        join=True,
        start_method="fork")

print(f"torch version : {torch.__version__}")

import torch._dynamo.config as config
print(f"is optimize_ddp_lazy_compile present? {hasattr(config, 'optimize_ddp_lazy_compile')}")