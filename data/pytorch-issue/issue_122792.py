"""
torchrun --standalone --nproc_per_node=2 repro_dcp_compile.py
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(4, 4)
        self.lin2 = nn.Linear(4, 4)
        self.register_buffer("buf", torch.randn((4,)), persistent=False)
        self.weight = nn.Parameter(torch.randn((4, 4)))


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    model = Model()
    model = torch.compile(model)

    sharded_sd = get_model_state_dict(model)
    set_model_state_dict(model, sharded_sd)