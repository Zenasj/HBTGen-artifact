import torch

dist.all_gather_into_tensor(
                output_tensor=flat_tensor, input_tensor=input_flat_tensor_shard, group=group
            )

dist.all_gather_into_tensor(
                flat_tensor, input_flat_tensor_shard, group=group
            )

dist.all_gather_into_tensor(
                flat_tensor, input_flat_tensor_shard, group=group
            )

def _traceable_collective_remaps():
    # We can't rely on importing from distributed, since it's not always built
    if torch.distributed.is_available():
        from torch.distributed._functional_collectives import (
            traceable_collective_remaps,
        )

        return traceable_collective_remaps
    return {}