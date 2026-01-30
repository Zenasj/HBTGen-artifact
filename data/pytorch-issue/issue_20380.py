import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    print(f"starting rank {rank}")
    dist.init_process_group(
        "gloo",
        init_method="tcp://localhost:12345",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=10)
    )
    print(f"started rank {rank}")
    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(setup, args=(2, ), nprocs=2, join=True)