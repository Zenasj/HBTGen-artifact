import torch.nn as nn

import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn

def worker(rank, streamed):
    # Set up the worker to use NCCL
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    
    # Pick the stream we're going to work on
    stream = torch.cuda.Stream() if streamed else torch.cuda.default_stream()
    with torch.cuda.stream(stream):

        # Create the network
        net = DDP(nn.Linear(1, 1, bias=False).cuda(rank), device_ids=[rank])

        # Do a bunch of forward-backward passes and check the gradients are 
        # consistent after each
        print(f'{rank}: Looping')
        bad = False
        for i in range(100):
            # Clear the gradients manually since we don't have an optimizer around
            grad = net.module.weight.grad
            if grad is not None:
                grad.detach_()
                grad.zero_()

            # Forward-backward. Gradient on worker #rank should be [[rank]].
            batch = torch.tensor([rank]).float().cuda(rank)
            loss = net(batch).sum()
            loss.backward()

            # Get the new gradient
            grad = net.module.weight.grad

            # Calculate the average gradient across workers. It should be .5.
            average = grad.clone()
            dist.all_reduce(average) # takes the sum across ranks
            average = average/2 

            if average[0, 0] != .5:
                print(f'{rank}: Average-of-averaged-gradients is wrong on loop {i}; it\'s {average[0, 0]} when my own averaged-gradient is {grad[0, 0]}. They should be both .5')
                bad = True

        if not bad:
            print(f'{rank}: Looped successfully; all grads consistent')

def run(streamed):
    # Launch a pair of workers that'll use DDP to compute gradients together
    workers = [mp.Process(target=worker, args=(r, streamed)) for r in [0, 1]]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

if __name__ == '__main__':
    mp.set_start_method('spawn')

    print('\nRunning with-stream, with-bug version')
    run(streamed=True)

    print('Running no-stream, no-bug version')
    run(streamed=False)