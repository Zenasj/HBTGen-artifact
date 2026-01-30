import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def debug_group(group):
    tensor = torch.tensor([10 ** dist.get_rank()], dtype=torch.float).cuda()
    dist.all_reduce(tensor, group=group)
    print(dist.get_rank(), tensor)


def par_main(rank: int, n: int, m: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "11355"
    dist.init_process_group("nccl", rank=rank, world_size=n * m)
    torch.cuda.set_device(rank)
    rank_mat = [list(range(i * n, (i + 1) * n)) for i in range(m)]
    row_group, _ = dist.new_subgroups_by_enumeration(rank_mat)
    col_group, _ = dist.new_subgroups_by_enumeration(list(map(list, zip(*rank_mat))))

    dist.barrier()
    debug_group(row_group)
    dist.barrier()
    debug_group(col_group)
    dist.barrier()
    dist.destroy_process_group()


def run(n=3, m=2):
    mp.spawn(par_main, args=(n, m), nprocs=n * m, join=True)


if __name__ == "__main__":
    run()