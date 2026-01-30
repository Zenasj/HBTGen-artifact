import torch.nn as nn

#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import gc
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch.multiprocessing import Process
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.utils.data.distributed import DistributedSampler


def show_gpu_memory(device=None):
    if device is None:
        device = torch.cuda.current_device()
    print('## GPU memory usage: rank={}, CurrentMem={:.2f}MiB, MaxMem={:.2f}MiB, CachedMem={:.2f}MiB'.format(
        device,
        torch.cuda.memory_allocated(device) / (1 << 20),
        torch.cuda.max_memory_allocated(device) / (1 << 20),
        torch.cuda.memory_cached(device) / (1 << 20)
    ), flush=True)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]


class Child(torch.nn.Module):
    def __init__(self, lin_in, linear, lin_out):
        super().__init__()
        self.lin_in = lin_in
        self.linear = linear
        self.lin_out = lin_out

    def forward(self, x):
        return self.lin_out(self.linear(self.lin_in(x)))


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(2000, 5000)     # 38.14MiB
            for _ in range(10)
        ])
        self.lin_in = torch.nn.Linear(3, 2000)
        self.lin_out = torch.nn.Linear(5000, 4)

    @property
    def num_linears(self):
        return len(self.linears)

    def get_child(self, i):
        return Child(self.lin_in, self.linears[i % self.num_linears], self.lin_out)


def get_dataset():
    data_list = [{"x": torch.FloatTensor(3).random_(), "y": torch.FloatTensor(4).random_()} for i in range(10)]
    return CustomDataset(data_list)


def train(args, rank, size, device_name):
    print('| rank={}, size={}, device_name={}'.format(rank, size, device_name))
    dataset = get_dataset()
    distributed_sampler = DistributedSampler(dataset, size, rank)
    train_loader = DataLoader(dataset, pin_memory=True, sampler=distributed_sampler)
    device = torch.device(device_name)
    model = Model()
    model.to(device)

    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    children = [None for _ in range(model.num_linears)]

    num_updates = 0
    for epoch in range(100):
        model.train()
        for batch_num, batch in enumerate(train_loader):
            print('epoch={}, rank={}, batch_num={}, x_shape={}'.format(
                epoch, rank, batch_num, batch["x"].shape))
            show_gpu_memory(device)
            optimizer.zero_grad()

            # Create child from model.
            if not args.cache_child:
                child = model.get_child(num_updates)
            else:
                child = children[num_updates % len(children)]
                if child is None:
                    child = model.get_child(num_updates)
                    children[num_updates % len(children)] = child

            # Parallel child
            if args.dp == 'ddp':
                child = DistributedDataParallel(child, device_ids=[device])
            elif args.dp == 'dp':
                child = DataParallel(child, device_ids=[device])
            elif args.dp == 'lddp':
                from legacy_distributed_data_parallel import LegacyDistributedDataParallel
                child = LegacyDistributedDataParallel(module=child, world_size=2)
            elif args.dp == 'none':
                pass
            else:
                raise ValueError('Unknown data-parallel model {!r}'.format(args.dp))

            pred = child(batch['x'].to(device))
            exp = batch['y'].to(device)
            loss = F.binary_cross_entropy_with_logits(exp, pred)
            loss.backward()
            del loss

            optimizer.step()
            num_updates += 1
    gc.collect()
    torch.cuda.empty_cache()
    print('Final GPU memory usage:')
    show_gpu_memory(device)


def init_processes(args, rank, size, fn, ddp_args, device):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = ddp_args['master_addr']
    os.environ['MASTER_PORT'] = ddp_args['master_port']
    os.environ['WORLD_SIZE'] = str(size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(ddp_args['backend'], rank=rank, world_size=size)
    fn(args, rank, size, device)
    print("done, rank={}".format(rank))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dp', choices=['dp', 'ddp', 'lddp', 'none'], default='ddp',
                        help='data-parallel model, default is %(default)r')
    parser.add_argument('--no-cache-child', action='store_false', dest='cache_child', default=True,
                        help='disable cache child')
    args = parser.parse_args()

    start_time = time.time()
    ddp_args = {"master_addr": "127.0.0.1", "master_port": "29500", "backend": "gloo"}
    devices = ["cuda:0", "cuda:1"]
    size = len(devices)
    processes = []
    for rank, device in enumerate(devices):
        p = Process(target=init_processes, args=(args, rank, size, train, ddp_args, device))
        p.start()
        processes.append(p)

    while not any(p.exitcode is not None for p in processes):  # all worker processes have finished
        time.sleep(0.2)

    for p in processes:
        print("join")
        p.join()
        print("join done")
    print("all done")

    print('Time passed: {:.6f}s'.format(time.time() - start_time))


if __name__ == "__main__":
    main()