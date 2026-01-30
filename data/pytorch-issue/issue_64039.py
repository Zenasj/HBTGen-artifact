import torch.nn as nn

py
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


class BatchNorm1dModel(torch.nn.Module):

    def __init__(self, feat_dim):
        super().__init__()
        self.bnorm = torch.nn.BatchNorm1d(feat_dim)
        self.linear1 = torch.nn.Linear(feat_dim, feat_dim)
        self.linear2 = torch.nn.Linear(feat_dim, feat_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bnorm(x.transpose(1,2)).transpose(1,2)
        x = self.linear2(x)
        return x


class BatchNorm2dModel(torch.nn.Module):

    def __init__(self, feat_dim):
        super().__init__()
        self.bnorm = torch.nn.BatchNorm2d(feat_dim)
        self.linear1 = torch.nn.Linear(feat_dim, feat_dim)
        self.linear2 = torch.nn.Linear(feat_dim, feat_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = x.unsqueeze(-1)
        x = self.bnorm(x.transpose(1,2)).transpose(1,2)
        x = x.squeeze(-1)
        x = self.linear2(x)
        return x


def main(rank):
    # iniialize the process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="gloo", rank=rank, world_size=2)

    torch.manual_seed(0)
    model_1d = BatchNorm1dModel(3).to(rank).double()

    torch.manual_seed(0)
    model_2d = BatchNorm2dModel(3).to(rank).double()

    inputs = torch.randn(2, 2, 3, requires_grad=True).double()

    for model in [model_1d, model_2d]:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])

        outputs = model(inputs)
        outputs.sum().backward()
        if rank == 0:
            print("OUTPUT\n", outputs)
            print("LINEAR1 GRAD\n", model.module.linear1.weight.grad)
            print("LINEAR2 GRAD\n", model.module.linear2.weight.grad, end="\n\n")

        torch.autograd.gradcheck(model, (inputs,))


if __name__ == "__main__":
    mp.spawn(main, nprocs=2, join=True)