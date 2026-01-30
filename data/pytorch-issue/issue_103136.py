with partial_state_only(model):
    sd = model.state_dict()

import contextlib
import os
from functools import partial

import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 50, bias=False)
        self.l2 = nn.Linear(50, 1, bias=False)


@contextlib.contextmanager
def partial_state_only(module: nn.Module):
    originals = {}

    def save(name, destination, prefix, keep_vars):
        if "l1" in prefix:
            original_fn = originals[name]
            return original_fn(destination, prefix, keep_vars)

    for name, submodule in module.named_modules():
        originals[name] = submodule._save_to_state_dict
        submodule._save_to_state_dict = partial(save, name)
    yield
    for name, module in module.named_modules():
        module._save_to_state_dict = originals[name]


def work(rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "1234"
    dist.init_process_group("nccl", world_size=1, rank=rank)

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    model = MyModel().to(device)
    model = FullyShardedDataParallel(model)

    state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    state_dict_type_context = FullyShardedDataParallel.state_dict_type(
        module=model,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=state_dict_config,
        optim_state_dict_config=optim_state_dict_config,
    )

    with partial_state_only(model), state_dict_type_context:
        sd = model.state_dict()
        print(sd)


def run():
    mp.spawn(work, nprocs=1)


if __name__ == "__main__":
    run()