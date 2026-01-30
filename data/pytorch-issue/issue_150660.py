import torch
import torch.distributed as dist
import os

def main():
    # Initialize the distributed process group using NCCL
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    # Create a tensor on the GPU with a value equal to the rank
    tensor = torch.tensor([rank], device=torch.device(f"cuda:{local_rank}"))
    
    # All-reduce: sum up the tensor values from all processes
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    print(f"Global Rank {rank}; Local Rank {local_rank} has tensor value: {tensor.item()}")

if __name__ == '__main__':
    main()

import torch
import torch.distributed as dist
import os
import socket

def get_sorted_hosts():
    master_ip = socket.gethostbyname(os.environ["MASTER_ADDR"])
    node_ips = {socket.gethostbyname(os.environ[var]) for var in os.environ if var.startswith("TORCHELASTIC_AGENT") and "ADDR" in var}
    return [master_ip] + sorted(node_ips - {master_ip})

def main():
    node_ips = get_sorted_hosts()
    current_ip = socket.gethostbyname(socket.gethostname())

    # Assign ranks based on sorted order (force master to rank 0)
    rank = node_ips.index(current_ip)
    world_size = len(node_ips)
    local_rank = int(os.environ["LOCAL_RANK"])

    # Set the correct rank
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    tensor = torch.tensor([rank], device=torch.device(f"cuda:{local_rank}"))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"Global Rank {rank}; Local Rank {local_rank} has tensor value: {tensor.item()}")

if __name__ == '__main__':
    main()

def get_sorted_hosts():
    master_ip = socket.gethostbyname(os.environ["MASTER_ADDR"])
    node_ips = {socket.gethostbyname(os.environ[var]) for var in os.environ if var.startswith("TORCHELASTIC_AGENT") and "ADDR" in var}
    return [master_ip] + sorted(node_ips - {master_ip})