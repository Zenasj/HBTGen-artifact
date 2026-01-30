import torch
import torch.distributed as dist
import argparse
import os


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default='127.0.0.1')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    with open(f'log_{rank}.txt', 'w') as f:
        print("master addr", master_addr)
        print("master port", master_port)
        print("rank", rank)
        f.write("master addr: ")
        f.write(master_addr)
        f.write("\nmaster port: ")
        f.write(master_port)

    dist.init_process_group(
        backend='nccl',
        init_method=f'env://',
    )

    counter = 0
    while(1):
        counter += 1
        if counter == 1000:
            print(f'rank: {rank} alive')
            counter = 0
    dist.destroy_process_group()

os.environ["MASTER_ADDR"] = RANK0_IP_OR_HOSTNAME_BEFORE_2ND_NODE_JOIN

os.environ["MASTER_ADDR"] = RANK0_IP_OR_HOSTNAME_AFTER_2ND_NODE_JOIN

hostname 
hostname -f