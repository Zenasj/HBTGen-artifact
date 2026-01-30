# test_gloo.py
import torch
import torch.distributed as dist
dist.init_process_group(backend="gloo")
groups = [[0, 4, 8, 12, 16, 20, 24, 28], [1, 5, 9, 13, 17, 21, 25, 29], [2, 6, 10, 14, 18, 22, 26, 30], [3, 7, 11, 15, 19, 23, 27, 31]]
rank = dist.get_rank()

self_group = None

for item in groups:
    if rank in item:
        self_group = item

gloo_group = dist.new_group(ranks=self_group, backend="gloo")
print(f"[{rank}]gloo_group finished.. rank {dist.get_rank(group=gloo_group)} size {dist.get_world_size(group=gloo_group)}")

for item in groups:
    gloo_group = dist.new_group(ranks=item, backend="gloo")
    if rank in item:
        self_group = gloo_group