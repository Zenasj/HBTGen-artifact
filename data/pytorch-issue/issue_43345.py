import torch
import os


def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10638"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"

    print(f"init nccl {rank}, {world_size}")
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method=init_method
    )

    if False: # enable to fix it
        torch.distributed.all_reduce(torch.zeros(1).cuda())

    assert torch.distributed.is_initialized()
    assert rank == torch.distributed.get_rank()
    assert world_size == torch.distributed.get_world_size()

    torch.cuda.set_device(0)

    print(f"dist init for {rank}")
    torch.distributed.barrier()

    print(f"past barrier for {rank}")
    t = torch.Tensor(range(10)).cuda()

    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    print(f"past all_reduce for {rank}")
    assert torch.equal(t, torch.Tensor(range(0, 20, 2)).cuda())

if __name__ == "__main__":
    world_size = 2
    torch.multiprocessing.spawn(run, args=(world_size,), nprocs=world_size, join=True)