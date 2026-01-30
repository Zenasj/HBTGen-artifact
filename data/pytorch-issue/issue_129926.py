import ray
import torch
import torch.distributed as dist
from ray.air.util.torch_dist import (
    TorchDistributedWorker,
    init_torch_dist_process_group,
    shutdown_torch_dist_process_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

@ray.remote(num_gpus=1)
class TestWorker(TorchDistributedWorker):
    def __init__(self):
        super().__init__()

    def run(self):
        rank = torch.distributed.get_rank()
        dev = f"cuda:{ray.get_gpu_ids()[0]}"
        tensor = torch.tensor([rank]).to(dev)

        # case 1: whole group, part workers => blocking
        if rank > 1:
            group = dist.new_group([0, 1, 2, 3])
            dist.broadcast(tensor, 2, group)

        # case 2: part group, all workers => success
        group = dist.new_group([2, 3])
        dist.broadcast(tensor, 2, group)

        # case 3: part group, part workers => success
        if rank > 1:
            group = dist.new_group([2, 3])
            dist.broadcast(tensor, 2, group)

        # case 4: different groups, all workers => error
        if rank < 2:
            group = dist.new_group([0, 1])
            dist.broadcast(tensor, 0, group)
        else:
            group = dist.new_group([2, 3])
            dist.broadcast(tensor, 2, group)

        return tensor

def run_workers():
    placement_group = ray.util.placement_group(
        [{"CPU": 1, "GPU": 1}] * 4,
        strategy="STRICT_PACK",
    )
    ray.get(placement_group.ready(), timeout=1000)

    workers = [
        TestWorker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group
            )
        ).remote()
        for _ in range(4)
    ]

    init_torch_dist_process_group(workers, backend="nccl")
    ray.get([w.run.remote() for w in workers])
    shutdown_torch_dist_process_group(workers)

if __name__ == "__main__":
    run_workers()