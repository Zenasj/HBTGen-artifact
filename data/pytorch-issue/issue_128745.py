import logging
import os
from typing import Callable, Optional, Tuple
from torch.distributed._tensor import DTensor

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.checkpoint import state_dict as ptd_state_dict
from torch.distributed._composable.fsdp import fully_shard
import pdb
import sys

def main():
    dist.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    with torch.device("meta"):
        meta_model = nn.Sequential(*[nn.Linear(4, 4, bias=False) for _ in range(2)])
        for layer in meta_model:
            fully_shard(layer)
        fully_shard(meta_model)
    with torch.device("cpu"):
        cpu_model = nn.Sequential(*[nn.Linear(4, 4, bias=False) for _ in range(2)])
        full_sd = cpu_model.state_dict()
    ptd_state_dict.set_model_state_dict(
        meta_model,
        model_state_dict=full_sd,
        # 'aten.copy_.default: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!',).
        options=ptd_state_dict.StateDictOptions(
            full_state_dict=True, strict=False
        )
        # NotImplementedError: c10d::broadcast_: attempted to run this operator with Meta tensors
        # options=ptd_state_dict.StateDictOptions(
        #     broadcast_from_rank0=True, full_state_dict=True, strict=False
        # )
    )
    
if __name__ == "__main__":
    main()