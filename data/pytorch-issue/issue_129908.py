import logging
import os
import time
import torch
import torch.distributed as dist


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    backend = 'nccl'
    group = torch.distributed.init_process_group(backend=backend)
    rank = torch.distributed.get_rank(group=group)

    for i in range(4):
        time.sleep(rank)
        logging.info(f"Rank {rank}: enter barrier {i}")
        dist.barrier()
        logging.info(f"Rank {rank}: exit barrier {i}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

import logging
import os
import time
import torch
import torch.distributed as dist


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    backend = 'nccl'
    group = torch.distributed.init_process_group(backend=backend)
    rank = torch.distributed.get_rank(group=group)

    for i in range(4):
        time.sleep(rank)
        logging.info(f"Rank {rank}: enter barrier {i}")
        dist.barrier()
        logging.info(f"Rank {rank}: exit barrier {i}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()