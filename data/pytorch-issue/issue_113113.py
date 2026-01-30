py
import os

import torch.cuda
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torchmetrics import Accuracy
from torch.distributed.fsdp import FullyShardedDataParallel


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        # This metric is an nn.Module, but doesn't have parameters
        self.metric = Accuracy(task="multiclass", num_classes=10)


def work(rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "1234"
    dist.init_process_group("nccl", world_size=2, rank=rank)

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    model = Model()
    model = FullyShardedDataParallel(mode, device_id=rank)

    assert not any(model.metric.parameters())

    assert model.linear.weight.device == device
    assert model.metric.tp.device == device  # fails!


def run():
    mp.spawn(work, nprocs=2)


if __name__ == "__main__":
    run()

assert model.linear.weight.device == device
AssertionError