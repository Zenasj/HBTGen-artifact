import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
import argparse


def setup(rank, world_size):
    torch.manual_seed(0)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # iniialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self, num_features=10):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.bn(x)
        x = x.permute(0, 2, 1, 3)
        return x


def demo_basic(rank, world_size, convert_sync_batchnorm):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel(num_features=10)
    # create model and move it to GPU with id rank
    if world_size > 1 and convert_sync_batchnorm:
        print("Converting BatchNorm to SyncBatchNorm")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(rank)
    if world_size > 1:
        ddp_model = DDP(model, device_ids=[rank])
        model = ddp_model

    inputs = torch.randn(2, 2, 10, 1).to(rank)
    print(f"{rank=}, {inputs[0, 0].squeeze()=}")
    outputs = model(inputs)
    print(f"{rank=}, {outputs[0, 0].squeeze()=}")

    cleanup()


def run_demo(demo_fn, world_size, convert_bn):
    if world_size > 1:
        mp.spawn(demo_fn, args=(world_size, convert_bn), nprocs=world_size, join=True)
    else:
        demo_fn(0, world_size, convert_bn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpus", default=2, type=int, help="Number of gpus")
    parser.add_argument(
        "--convert_bn",
        action="store_true",
        default=False,
        help="Whether to convert BatchNorm layers to SyncBatchNorm",
    )

    args = parser.parse_args()

    print(f"{torch.__version__=}")
    run_demo(demo_basic, args.n_gpus, args.convert_bn)

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
import argparse


def setup(rank, world_size):
    torch.manual_seed(0)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # iniialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self, num_features=10):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x):
        x = self.bn(x)
        return x


def demo_basic(rank, world_size, convert_sync_batchnorm, permute):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel(num_features=3)
    # create model and move it to GPU with id rank
    if world_size > 1 and convert_sync_batchnorm:
        print("Converting BatchNorm to SyncBatchNorm")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(rank)
    if world_size > 1:
        ddp_model = DDP(model, device_ids=[rank])
        model = ddp_model

    inputs = torch.randn(2, 3, 3, 1).to(rank)
    if permute:
        inputs = inputs.permute(0,2,1,3)
    print(f"{rank=}, {inputs[0].squeeze()=}")
    outputs = model(inputs)
    print(f"{rank=}, {outputs.squeeze()=}")

    cleanup()


def run_demo(demo_fn, world_size, convert_bn, permute):
    if world_size > 1:
        mp.spawn(demo_fn, args=(world_size, convert_bn, permute), nprocs=world_size, join=True)
    else:
        demo_fn(0, world_size, convert_bn, permute)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpus", default=2, type=int, help="Number of gpus")
    parser.add_argument(
        "--convert_bn",
        action="store_true",
        default=False,
        help="Whether to convert BatchNorm layers to SyncBatchNorm",
    )
    parser.add_argument(
        "--permute",
        action="store_true",
        default=False,
        help="Whether to convert BatchNorm layers to SyncBatchNorm",
    )
    args = parser.parse_args()

    print(f"{torch.__version__=}")
    run_demo(demo_basic, args.n_gpus, args.convert_bn, args.permute)