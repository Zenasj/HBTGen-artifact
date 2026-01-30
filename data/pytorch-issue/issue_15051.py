py
import torch
torch.cuda.set_device(0)
torch.distributed.init_process_group("NCCL", init_method="file://...", rank=0, world_size=2)
torch.distributed.barrier()

py
import torch
torch.cuda.set_device(0)
torch.distributed.init_process_group("NCCL", init_method="file://...", rank=1, world_size=2)
torch.distributed.barrier()

py
x = torch.tensor(torch.distributed.get_rank(), dtype=torch.float).cuda()
ys = [torch.empty(()).cuda(), torch.empty(()).cuda()]
print(ys)  # <- hang

def barrier():
    t = torch.randn((), device='cuda')
    dist.all_reduce(t)
    torch.cuda.synchronize()