from torch.multiprocessing import spawn
import torch.distributed as dist
import torch
import os
def test(rank, world_size):
    # Code runs on each rank.
    dist.init_process_group("nccl", rank=rank, world_size=2)
    output = torch.ones((1000, 1000)).cuda(rank)
    s = torch.cuda.Stream()
    handle = dist.all_reduce(output, async_op=True)
    # Wait ensures the operation is enqueued, but not necessarily complete.
    # Using result on non-default stream.
    with torch.cuda.stream(s):
        handle.wait()
        # s.wait_stream(torch.cuda.default_stream())
        output.add_(100)
        if rank == 0:
            print(output)
    

    dist.destroy_process_group()
    
if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29503'
    spawn(test, args=(2,), nprocs=2, join=True)