import os
import torch
import torch.distributed as dist

def worker(rank, world_size):
    # Initialize the process group without specifying the backend
    dist.init_process_group(backend=None, rank=rank, world_size=world_size)
    
    # Set the CUDA device for the current process based on the rank (only for ranks < 4)
    if rank < 4:
        torch.cuda.set_device(rank % torch.cuda.device_count())
    
    # Create an NCCL process group, including ranks 0, 1, 2, 3
    if rank < 4:
        group_nccl = dist.new_group(ranks=[0, 1, 2, 3], backend="nccl")
    
    # Create a GLOO process group, including ranks 3, 4
    if rank >= 3:
        group_gloo = dist.new_group(ranks=[3, 4], backend="gloo")
    
    if rank == 0:
        tensor = torch.tensor([1.0]).cuda()
        dist.send(tensor=tensor, dst=1, group=group_nccl)
        print(f"Rank {rank} sent tensor {tensor} to Rank 1 using NCCL")
    elif rank == 1:
        tensor = torch.tensor([0.0]).cuda()
        dist.recv(tensor=tensor, src=0, group=group_nccl)
        print(f"Rank {rank} received tensor {tensor} from Rank 0 using NCCL")
        dist.send(tensor=tensor, dst=2, group=group_nccl)
        print(f"Rank {rank} sent tensor {tensor} to Rank 2 using NCCL")
    elif rank == 2:
        tensor = torch.tensor([0.0]).cuda()
        dist.recv(tensor=tensor, src=1, group=group_nccl)
        print(f"Rank {rank} received tensor {tensor} from Rank 1 using NCCL")
        dist.send(tensor=tensor, dst=3, group=group_nccl)
        print(f"Rank {rank} sent tensor {tensor} to Rank 3 using NCCL")
    elif rank == 3:
        tensor_nccl = torch.tensor([0.0]).cuda()
        dist.recv(tensor=tensor_nccl, src=2, group=group_nccl)
        print(f"Rank {rank} received tensor {tensor_nccl} from Rank 2 using NCCL")

        tensor_gloo = tensor_nccl.cpu()
        dist.send(tensor=tensor_gloo, dst=4, group=group_gloo)
        print(f"Rank {rank} sent tensor {tensor_gloo} to Rank 4 using GLOO")
    elif rank == 4:
        tensor = torch.tensor([0.0])
        dist.recv(tensor=tensor, src=3, group=group_gloo)
        print(f"Rank {rank} received tensor {tensor} from Rank 3 using GLOO")

def main():
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    worker(rank, world_size)

if __name__ == "__main__":
    main()