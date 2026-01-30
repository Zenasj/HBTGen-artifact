import torch.nn as nn

import torch
import torch.distributed as dist
import torch.distributed.nn
from functools import partial
def worker(gpu, USE_NN_REDUCE = 0):
    dist.init_process_group(
        backend="nccl", init_method="tcp://localhost:12345", world_size=2, rank=gpu
    )
    torch.cuda.set_device(gpu)
    torch.manual_seed(gpu)
    torch.cuda.manual_seed_all(gpu)
    x = torch.randn((2, 2, 3), device='cuda', requires_grad=True)
    xx = torch.nn.functional.normalize(x, p=2, dim=-1)
    cov = torch.stack([W.T.matmul(W) for W in xx])
    if USE_NN_REDUCE:
        cov=dist.nn.all_reduce(cov)
    else:
        dist.all_reduce(cov)
    print("Value after all_reduce:", cov)
    y = torch.logdet(torch.ones((3,3), device=gpu)+ 0.1*cov).sum()
    #y = sum([torch.logdet(torch.ones((3,3), device=gpu)+ 0.1*cov[i]) for i in range(2)])
    y.backward()
    print(f"{USE_NN_REDUCE=}, {gpu=}, {y=}, {x.grad=}")

nn_worker = partial(worker, USE_NN_REDUCE=1)
def local():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    x0 = torch.randn((2, 2, 3), device='cuda')
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    x1 =  torch.randn((2, 2, 3), device='cuda')
    x = torch.cat([x0, x1], dim=1)
    x = x.requires_grad_()
    xx = torch.nn.functional.normalize(x, p=2, dim=-1)
    xx_all_reduce = torch.stack([W.T.matmul(W) for W in xx])
    print(f"truth: xx_all_reduce={xx_all_reduce}")
    y = torch.logdet(torch.ones((3,3), device='cuda')+ 0.1*xx_all_reduce).sum()
    y.backward()
    print(f"truth: {y=}")
    print(f"truth: grad={x.grad}")

if __name__ == "__main__":
    #dist.init_process_group(backend="nccl")
    # if dist.get_rank()==0:
    #     local()
    # worker(dist.get_rank())
    # nn_worker(dist.get_rank())
    local()
    torch.multiprocessing.spawn(worker, nprocs=2)
    torch.multiprocessing.spawn(nn_worker, nprocs=2)