import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta

def main(rank, world_size):
    if rank == 0:
        print("creating store")
        # world size is 2 so this eventually times out
        store = dist.TCPStore("localhost", 1234, 2, True, timeout=timedelta(seconds=5))
        print("finished creating store")

if __name__ == "__main__":
    world_size = 2
    mp.spawn(main, (world_size,), nprocs=world_size)