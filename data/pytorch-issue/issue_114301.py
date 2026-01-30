import torch

def run_worker(rank, world_size):
    ndev = torch.cuda.device_count()
    device = torch.device(f'cuda:{rank % ndev}')
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=20)
                           )
    print(torch.distributed.get_world_size())
    pg = c10d.distributed_c10d._get_default_group()

    device_id = f"cuda:{rank % torch.cuda.device_count()}"
    tensor0 = torch.ones([1]).cuda(rank) * rank
    tensor = torch.ones([1]).cuda(rank) * (rank+1)
    # for initialization purpose
    torch.distributed.all_reduce(tensor0, group=pg)    
    print(tensor0.item())
    # suppose before the following operations, one rank is down.
    if rank==2:
        time.sleep(100)
    print("One rank is down, other ranks continue")
    from torch.distributed.distributed_c10d import _coalescing_manager
    with _coalescing_manager(pg, device, async_ops=True) as cm:
        for i in [tensor]:
            torch.distributed.all_reduce(
                i, op=torch.distributed.ReduceOp.AVG, group=pg, async_op=True,
            )
        cm.wait()
    print("end")