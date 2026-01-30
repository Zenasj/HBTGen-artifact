import torch
import torch.nn as nn

import functools, os, torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import always_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing_wrapper,
)

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer0 = torch.nn.Linear(6, 6)
        self.layer1 = torch.nn.Linear(6, 6, bias=False)

    def forward(self, x):
        z = self.layer0(x)
        z = self.layer1(z)
        return z

    def get_input(self, device: torch.device):
        return (torch.randn((8, 6)).to(device),)

    def get_loss(self, input, output):
        return (output - input[0]).sum()


def fsdp_main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group("nccl")
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
    model = FSDP(
        Model(),
        auto_wrap_policy=always_wrap_policy,
        backward_prefetch=None,
        forward_prefetch=False,
        device_id=torch.cuda.current_device(),
    )
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, FSDP)
    apply_activation_checkpointing_wrapper(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )
    if local_rank == 0:
        print(model)
    input = model.module.get_input(local_rank)
    output = model(*input)
    loss = model.module.get_loss(input, output).to(local_rank)
    loss.backward()


if __name__ == "__main__":
    fsdp_main()