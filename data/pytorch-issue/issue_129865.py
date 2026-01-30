import torch
import torch.distributed as dist
import time
import os

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(backend, rank=rank, world_size=size, device_id=device)
    fn(rank, size)

def traditional_new_group(ranks):
    """ Traditional method to create a new process group. """
    new_group = dist.new_group(ranks=ranks)
    return new_group

def optimized_new_group(ranks):
    """ Optimized method to create a new process group. """
    # Hypothetical API call to create a new group by reusing the existing process group
    new_group = dist.new_group(ranks=ranks)
    return new_group

def benchmark(rank, size):
    """ Benchmark the time to create new process groups and broadcast tensors. """
    iterations = 100
    tensor = torch.zeros(1).cuda(rank)
    
    # Traditional method
    start_time = time.time()
    start_time = time.time()
    n_list=[0,1,2,3]
    for _ in range(iterations):
        group = optimized_new_group(n_list)
        if rank in n_list:
            dist.broadcast(tensor, src=0, group=group)
    optimized_time = (time.time() - start_time)/iterations
    
    if rank == 0:
        print(f"Optimized method time: {optimized_time:.6f} seconds")

if __name__ == "__main__":
    size = 4  # Number of processes
    processes = []
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    for rank in range(size):
        p = torch.multiprocessing.Process(target=init_process, args=(rank, size, benchmark))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()