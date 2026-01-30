import copy
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP

BACKEND = "nccl"
WORLD_SIZE = 2
BATCH_SIZE = 32

def worker(rank, world_size):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(BACKEND, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = torchvision.models.resnet152(pretrained=False)
    ddp_model1 = DDP(model.to(rank), device_ids=[rank], gradient_as_bucket_view=True)
    ddp_model2 = copy.deepcopy(ddp_model1)

    input = torch.randn(BATCH_SIZE, 3, 224, 224).to(rank)
    labels = torch.randn(BATCH_SIZE, 1000).to(rank)
    loss_fn = nn.MSELoss()

    grads1 = []
    output1 = ddp_model1(input)
    loss1 = loss_fn(output1, labels)
    loss1.backward()
    torch.cuda.synchronize()
    for p in ddp_model1.parameters():
        grads1.append(p.grad.clone())

    grads2 = []
    output2 = ddp_model2(input)
    loss2 = loss_fn(output2, labels)
    loss2.backward()
    torch.cuda.synchronize()
    for p in ddp_model2.parameters():
        grads2.append(p.grad.clone())

    if rank == 0:
        assert torch.allclose(output1, output2)
        for i, (g1, g2) in enumerate(zip(grads1, grads2)):
            if not torch.allclose(g1, g2):
                diff = torch.max(abs(g1 - g2)).item()
                if diff > 1e-5:
                    print(f"Significant mismatch on grad {i}: {diff:.8f}")

def main():
    world_size = WORLD_SIZE
    mp.spawn(worker,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()