import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP

print("ENSURE that you have a machine with 2 nvidia GPU's")

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "8080"
os.environ["WORLD_SIZE"] = "2"
os.environ["RANK"] = str(0)

def dpp_toy(rank, world_size):
    # create default process group
    print("Start example " + str(rank))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(rank)
    torch.cuda.set_device(device)

    # create local model
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.BatchNorm1d(10),
        nn.Linear(10, 10),
    ).to(device)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for i in range((rank + 1) * 10):
        signal = torch.tensor([1]).to(device)
        dist.all_reduce(signal)
        print(device)
        if signal.item() < world_size:
            break
        print("signal.item() did not hang")
        # forward pass
        outputs = ddp_model(torch.randn(20, 10).to(rank))
        labels = torch.randn(20, 10).to(rank)
        # backward pass
        loss_fn(outputs, labels).backward()
        # update parameters
        optimizer.step()
        print("Step " + str(i) + "_" + str(rank))

    if signal.item() >= world_size:
        dist.all_reduce(torch.tensor([0]).to(device))

    dist.barrier()
    print(f"{rank} done")


def dp_toy():
    # create default process group
    rank = 0

    # create local model
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.BatchNorm1d(10),
        nn.Linear(10, 10),
    )
    # construct DDP model
    dp_model = DP(model, device_ids=[0, 1]).to(rank)
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for i in range((rank + 1) * 10):
        outputs = dp_model(torch.randn(20, 10).to(rank))
        labels = torch.randn(20, 10).to(rank)
        # backward pass
        loss_fn(outputs, labels).backward()
        # update parameters
        optimizer.step()
        print("Step " + str(i))

    print(f"{rank} done")


def main():
    world_size = 2
    mp.spawn(dpp_toy,
        args=(world_size,),
        nprocs=world_size,
        join=True)

    #dp_toy()

if __name__=="__main__":
    main()

if signal.item() >= world_size:
        dist.all_reduce(torch.tensor([0]))  # <---

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP


def dp_toy():
    rank = 0

    # create local model
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.BatchNorm1d(10),
        nn.Linear(10, 10),
    )
    # construct DDP model
    dp_model = DP(model, device_ids=[0, 1]).to(rank)
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(dp_model.parameters(), lr=0.001)

    for i in range(10):
        print("Attempting to put data on gpu..")
        input_data = torch.randn(20, 10).to(rank)
        print("Attempting to call model..")
        outputs = dp_model(input_data)
        print("Model called")
        labels = torch.randn(20, 10).to(rank)
        # backward pass
        loss_fn(outputs, labels).backward()
        # update parameters
        optimizer.step()
        print("Step " + str(i))

    print(f"{rank} done")


def main():
    dp_toy()

if __name__=="__main__":
    main()

GRUB_CMDLINE_LINUX="iommu=soft"