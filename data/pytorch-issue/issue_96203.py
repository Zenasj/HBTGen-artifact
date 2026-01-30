# python -m torch.distributed.launch --nproc_per_node=2 this_file.py
import torch
from torchvision import datasets, transforms
import os
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_dim, empty_grad=False):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.BatchNorm1d = torch.nn.BatchNorm1d(hidden_dim)
        if empty_grad:
            self.layers2 = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim)])
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden = x
        hidden = self.linear(hidden)
        hidden = self.BatchNorm1d(hidden)

        return self.cross_entropy_loss(hidden, y)

def get_data_loader(model, total_samples, hidden_dim, device):
    batch_size = 8
    train_data = torch.randn(total_samples, hidden_dim, device=device).half()
    train_label = torch.empty(total_samples,
                              dtype=torch.long,
                              device=device).random_(hidden_dim)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=sampler)
    return train_loader

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    hidden_dim = 4
    device_id = rank % torch.cuda.device_count()
    model = SimpleModel(hidden_dim, empty_grad=False).to(device_id)
    model = model.half()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = DDP(model, device_ids=[device_id]).half()

    data_loader = get_data_loader(model=ddp_model,
                                  total_samples=1000,
                                  hidden_dim=hidden_dim,
                                  device=ddp_model.device)

    for n, batch in enumerate(data_loader):
        loss = ddp_model(batch[0], batch[1])
        if dist.get_rank() == 0:
            print("LOSS:", loss.item())

        if n == 1: break

if __name__ == "__main__":
    main()