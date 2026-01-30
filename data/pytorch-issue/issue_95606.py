import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank: int, world_size: int) -> None:
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, init_method='tcp://{}'.format('127.0.0.1:23456'), rank=rank, world_size=world_size)


def cleanup() -> None:
    dist.destroy_process_group()


def demo_basic(rank: int, queue: mp.JoinableQueue, world_size: int) -> None:
    setup(rank, world_size)
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'

    while True:
        batch = queue.get()
        batch = batch.to(device)

        try:
            negative_in_batch = batch.lt(0).all().item()
            if negative_in_batch:
                print("Found negative in batch", sys.stderr)

        finally:
            queue.task_done()


def split_batch(batch: torch.Tensor, world_size: int) -> torch.Tensor:
    return torch.split(batch, batch.shape[0] // world_size) # if I use torch.split(batch, batch.shape[0] // world_size).clone() instead no error is observed


def run_demo(world_size: int) -> None:
    print(torch.__version__, file=sys.stderr)
    num_batches = 10000
    batch_size = 64

    ctx = mp.get_context('spawn')
    queues = [ctx.JoinableQueue() for _ in range(world_size)]
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    processes = [ctx.Process(target=demo_basic, args=(i, queues[i], world_size)) for i in range(world_size)]

    for p in processes:
        p.start()

    for i in range(num_batches):
        large_batch = torch.randint(100000, size=(batch_size,))
        batches = split_batch(large_batch, world_size) # if I remove this line and send the large batch instead no error is observed
        print(f'queuing batch {i}', file=sys.stderr)

        for batch, queue in zip(batches, queues):
            queue.put(batch)

        for q in queues:
            q.join()

    for p in processes:
        p.terminate()


def main() -> None:
    run_demo(4)


if __name__ == '__main__':
    main()

import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def demo_basic(rank: int, queue: mp.JoinableQueue, world_size: int) -> None:
    dist.init_process_group(backend='gloo', init_method='tcp://{}'.format('127.0.0.1:23456'), rank=rank, world_size=world_size)
    while True:
        batch = queue.get()
        assert(batch.sum() == batch.numel())
        queue.task_done()

def split_batch(batch: torch.Tensor, world_size: int) -> torch.Tensor:
    # The .clone() is so there are no views to confuse the issue:
    return [x.clone() for x in torch.split(batch, batch.shape[0] // world_size)]

def run_demo(world_size: int) -> None:
    print(torch.__version__, file=sys.stderr)
    num_batches = 10000
    large_batch_size = 64

    ctx = mp.get_context('spawn')
    queues = [ctx.JoinableQueue() for _ in range(world_size)]
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    processes = [ctx.Process(target=demo_basic, args=(i, queues[i], world_size)) for i in range(world_size)]

    for p in processes:
        p.start()

    for i in range(num_batches):
        large_batch = torch.ones(large_batch_size)
        batches = split_batch(large_batch, world_size)

        for batch, queue in zip(batches, queues):
            #batch.share_memory_() # This line would prevent the crashes
            queue.put(batch)
            assert(batch.sum() == batch.numel()) # trying to use batch when it might be in the middle of being moved.

        for q in queues:
            q.join()

    for p in processes:
        p.terminate()

if __name__ == '__main__':
    run_demo(4)
    print('Finished without crashing.')