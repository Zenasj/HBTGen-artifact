import torch

# `gpu` refers to the current rank.

a = torch.zeros(1, 5).cuda(gpu).half() + gpu
print(f"Tensor on rank {gpu}: {a}")

gather_list = [torch.zeros(1, 5).cuda(gpu) for _ in range(dist.get_world_size())]
dist.all_gather(gather_list, a)
if gpu == 0:
    print(f"Collected tensor on rank 0:")
    for i in range(dist.get_world_size()):
        print(f"From GPU {i}: {gather_list[i]}")