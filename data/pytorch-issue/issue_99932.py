model = FullyShardedDataParallel(model,
                                 auto_wrap_policy=_auto_wrap_policy,
                                 device_id=torch.cuda.current_device(),
                                 )

for module in model.modules():
    if isinstance(module, FullyShardedDataParallel):
        module.clip_grad_norm_(max_norm=0.1197)

import os
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.api import ShardingStrategy
import torch.multiprocessing as mp

import logging

DEVICE_TYPE="cuda"


def setup(rank, world_size, use_cuda=True):
    logging.getLogger().setLevel(logging.DEBUG if rank == 0 else logging.CRITICAL)

    print(f"init for rank {rank}")
    if use_cuda:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=5))
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=5))

    # set device for nccl pg for collectives
    if use_cuda == "nccl":
        print(f"--> init device for rank {rank}")
        torch.cuda.set_device(rank)
    print (f"finished init for rank {rank}")

def _auto_wrap_policy(module: torch.nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
    if recurse:
        return True
    if hasattr(module, '_fsdp_wrap'):
        return bool(module._fsdp_wrap)
    return False

def simple_model_with_grads():
    # Set up small NN with one linear layer with no bias + softmax, so only
    # one set of params and get some gradients.
    N, hin, num_classes = 8, 4, 3
    x = torch.rand((N, hin))
    y = torch.randint(high=num_classes - 1, size=(N,))
    model = nn.Sequential(nn.Linear(hin, num_classes, bias=False), nn.Softmax(dim=1))

    # Force wrap every module in FSDP, to allow for testing FSDP
    # gradient clipping properly.
    for module in model.modules():
        module._fsdp_wrap = True

    model._fsdp_wrap = True
    model = FullyShardedDataParallel(model,
                                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                                    auto_wrap_policy=_auto_wrap_policy,
                                    device_id=torch.cuda.current_device(),
                                    )

    o = model(x)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(o, y)
    loss.backward()
    return model


def work_main(rank):
    model = simple_model_with_grads()

    for module in model.modules():
        if isinstance(module, FullyShardedDataParallel):
            module.clip_grad_norm_(max_norm=0.1197)


def main(rank, world_size, use_cuda=True):
    setup(rank, world_size, use_cuda)
    work_main(rank)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "29506"
    world_size = 2
    use_cuda = DEVICE_TYPE == "cuda"
    print(f"use_cuda == {use_cuda}")
    process = mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    process.join()