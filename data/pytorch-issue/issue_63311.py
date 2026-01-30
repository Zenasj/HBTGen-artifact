import random
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader


class RandomDataset(Dataset):

    def __getitem__(self, ind):
        item = [ind, random.randint(1, 10000)]
        return item

    def __len__(self):
        return 20


def worker(rank, main_func, world_size, dist_url):
    dist.init_process_group(backend="nccl", init_method=dist_url, world_size=world_size, rank=rank)
    dist.barrier()
    main_func()


def launch(main_func, start_method="fork"):
    port = 12123  # set your own port if provided port is not available
    dist_url = f"tcp://127.0.0.1:{port}"
    world_size = 2

    mp.start_processes(
        worker,
        nprocs=world_size,
        args=(main_func, world_size, dist_url),
        daemon=False,
        start_method=start_method,
    )

def main():
    loader = DataLoader(RandomDataset(), 2, shuffle=False, num_workers=4)
    rank = dist.get_rank()
    for batch in loader:
        print(f"rank{rank}: {batch}")
        dist.barrier()


if __name__ == "__main__":
    print("Start by fork:")
    launch(main, start_method="fork")
    print("Start by spawn:")
    launch(main, start_method="spawn")