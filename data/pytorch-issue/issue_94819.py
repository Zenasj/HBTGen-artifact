import os
import time
from argparse import ArgumentParser

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


SIZE = 10000


def run_worker(rank, use_nccl):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    if use_nccl:
        dist.init_process_group("nccl", world_size=2, rank=rank)
        dev0 = "cuda:0"
        dev1 = "cuda:1"
    else:
        dist.init_process_group("gloo", world_size=2, rank=rank)
        dev0 = "cpu"
        dev1 = "cpu"

    if rank == 0:
        ones = torch.ones(SIZE, SIZE, device=dev0)
        zeros = torch.zeros(SIZE, SIZE, device=dev0)

        print(f"[Main] tag 1: {ones.sum()}, tag 2: {zeros.sum()}")

        start_time = time.time()

        print(f"[Main] first send ones (tag 1) at {time.time() - start_time:5.3f} sec")
        f1 = dist.isend(ones, 1, tag=1)

        print(f"[Main] sleeps 3 sec at {time.time() - start_time:5.3f} sec")
        time.sleep(3)

        print(f"[Main] second send zeros (tag 2) at {time.time() - start_time:5.3f} sec")
        f2 = dist.isend(zeros, 1, tag=2)

        print(f"[Main] waiting at {time.time() - start_time:5.3f} sec")
        f2.wait()
        f1.wait()

    else:
        buf1 = torch.rand(SIZE, SIZE, device=dev1)
        buf2 = torch.rand(SIZE, SIZE, device=dev1)

        start_time = time.time()

        print(f"[Worker] first recv (tag 2) at {time.time() - start_time:5.3f} sec")
        f2 = dist.irecv(buf2, 0, tag=2)

        print(f"[Worker] second recv (tag 1) at {time.time() - start_time:5.3f} sec")
        f1 = dist.irecv(buf1, 0, tag=1)

        print(f"[Worker] waiting at {time.time() - start_time:5.3f} sec")

        f1.wait()
        print(f"[Worker] got tag 1: {buf1.sum()} at {time.time() - start_time:5.3f} sec")

        f2.wait()
        print(f"[Worker] got tag 2: {buf2.sum()} at {time.time() - start_time:5.3f} sec")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nccl", action="store_true")
    args = parser.parse_args()

    print(f"Use NCCL: {args.nccl}")

    mp.spawn(run_worker, nprocs=2, join=True, args=(args.nccl,))