import os

import torch
from torch import multiprocessing as mp
from torch.distributed import rpc
from torch.distributed.rpc import TensorPipeRpcBackendOptions


def add(x, y):
    print(x)  # tensor([1., 1.], device='cuda:1')
    return x + y, (x + y).to(2)


class Adder:
    def __init__(self):
        pass

    def add(self, x, y):
        return add(x, y)


def worker0():
    options = TensorPipeRpcBackendOptions(
        num_worker_threads=8,
        device_maps={"worker1": {0: 1}}
        # maps worker0's cuda:0 to worker1's cuda:1
    )
    options.set_device_map("worker1", {1: 2})
    # maps worker0's cuda:1 to worker1's cuda:2

    rpc.init_rpc(
        "worker0",
        rank=0,
        world_size=2,
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=options
    )

    x = torch.ones(2)
    rets = rpc.rpc_sync("worker1", add, args=(x.to(0), 1))
    # The first argument will be moved to cuda:1 on worker1. When
    # sending the return value back, it will follow the invert of
    # the device map, and hence will be moved back to cuda:0 and
    # cuda:1 on worker0
    print(rets[0])  # tensor([2., 2.], device='cuda:0')
    print(rets[1])  # tensor([2., 2.], device='cuda:1')

    print('so far so good')
    ####################################################
    x = torch.ones(2)
    adder_rref = rpc.remote("worker1", Adder)
    print('rref created successfully')
    rets = adder_rref.rpc_sync().add(x.to(0), 1)
    print(rets[0])  # tensor([2., 2.], device='cuda:0')
    print(rets[1])  # tensor([2., 2.], device='cuda:1')
    ####################################################
    rpc.shutdown()


def worker1():
    options = TensorPipeRpcBackendOptions(
        num_worker_threads=8,
    )

    rpc.init_rpc(
        "worker1",
        rank=1,
        world_size=2,
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=options
    )

    rpc.shutdown()


def run(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if rank == 0:
        worker0()
    else:
        worker1()


def main():
    mp.start_processes(run, nprocs=2, start_method='fork')


if __name__ == '__main__':
    main()