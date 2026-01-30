import torch

def main():
    # torchelastic sets the following env vars so that init_method="env://" works:
    #    RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    torch.distributed.init_process_group(init_method="env://")
    
    # run collective operations using the process group
    torch.distributed.all_reduce(...)

def main_rpc_backend_process_group():  
    rank=int(os.environ["RANK"]),
    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=int(os.environ["WORLD_SIZE"]),
        backend=BackendType.PROCESS_GROUP)
    # default process group is initialized as part of init_rpc()
    dist.all_reduce(...)