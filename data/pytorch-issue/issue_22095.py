import torch.nn as nn

import os
import argparse

import torch as th

th.autograd.set_detect_anomaly(True)

class Mod(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = th.nn.Linear(256, 32)
        self.register_buffer("buf", th.zeros(1, 32))

        # If buf is a Parameter instead of a buffer, there will be no inplace
        # error.
        #self.buf = th.nn.Parameter(th.zeros(1, 32))

    def forward(self, z):
        return (self.lin(z) * self.buf).sum()

        # If this line is substituted for the one above, there will be no
        # inplace error.
        #return (self.lin(z) * self.buf.clone()).sum()

def run(distributed=False, local_rank=None):
    mod = Mod().cuda()
    optim = th.optim.Adam(mod.parameters(), lr=1e-3)

    if distributed:
        mod = th.nn.parallel.DistributedDataParallel(mod,
                device_ids=[local_rank], output_device=local_rank)

    z1 = th.zeros(7, 256).cuda()
    z2 = th.zeros(7, 256).cuda()
    out = mod(z1)

    mod(z2) # <<---- The presence of this unused call causes an inplace error in backward() below if dec is a DDP module.

    loss = (out**2).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()

    print("Stepped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 0))
    args.distributed = world_size >= 1
    if args.distributed:
        print(f"[Main] Distributed across {world_size} GPUs")
        th.cuda.set_device(args.local_rank)
        th.distributed.init_process_group(backend="nccl", init_method="env://")

    run(distributed=args.distributed, local_rank=args.local_rank)