import torch.nn as nn

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._dynamo import compiled_autograd
import argparse
import os


world_size = 1
group = [i for i in range(world_size)]


def print_rank0(str):
    if dist.get_rank() == 0:
        print(str)


def grad_sync_clear(param, keep_grad):
    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    if not keep_grad:
        param.grad = None


class Module(torch.nn.Module):
    def __init__(self, ioc):
        super().__init__()
        self.fc1 = torch.nn.Linear(ioc, ioc, bias=False)
        self.fc2 = torch.nn.Linear(ioc, ioc, bias=False)
        self.fc3 = torch.nn.Linear(ioc, ioc, bias=False)
        self.fc4 = torch.nn.Linear(ioc, ioc, bias=False)

        self.grad_acc_hooks = []
        self.grad_acc = []
        self.params = [self.fc1.weight, self.fc2.weight,
                       self.fc3.weight, self.fc4.weight]
        for i, param in enumerate(self.params):

            keep_grad = False
            if i == dist.get_rank():
                keep_grad = True

            def wrapper(param, keep_grad):
                param_tmp = param.expand_as(param)
                grad_acc = param_tmp.grad_fn.next_functions[0][0]

                def grad_acc_hook(*notneeded):
                    grad_sync_clear(param, keep_grad)

                self.grad_acc.append(grad_acc)
                self.grad_acc_hooks.append(
                    grad_acc.register_hook(grad_acc_hook))

            wrapper(param, keep_grad)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x.sum()


def compiler_fn(gm):
    return torch.compile(gm, backend="inductor", fullgraph=True)


def run(rank):
    bs = 64
    ioc = 128

    model = Module(ioc)
    model_to_train = torch.compile(model, backend="inductor")

    input = torch.randn([bs, ioc])
    loss = model_to_train(input)
    with compiled_autograd.enable(compiler_fn):
        loss.backward()


def init_process(size, rank, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ["WORLD_SIZE"] = str(size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank)


if __name__ == "__main__":
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(
            world_size, rank, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()