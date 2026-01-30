import torch
import torch.nn as nn

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # The GRU or LSTM gets additional processes on GPU 0.
    ToyModel = nn.GRU(10, 10, 1)
    # The Linear does not get these problems.
    # ToyModel = nn.Linear(10,1)
    model = ToyModel.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    pbar_len = int(1e10 / 2)
    for _ in range(pbar_len):
        input_seq = torch.randn(4, 20,10)
        input_seq = input_seq.float().to(rank)
        ddp_model(input_seq)
    dist.destroy_process_group()
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    run_demo(demo_basic, world_size)

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    ToyModel = nn.GRU(10, 10, 1)
    model = ToyModel.cuda()
    ddp_model = DDP(model, device_ids=[rank])
    pbar_len = int(1e10 / 2)
    for _ in range(pbar_len):
        input_seq = torch.randn(4, 20,10)
        input_seq = input_seq.float().cuda()
        ddp_model(input_seq)
    dist.destroy_process_group()