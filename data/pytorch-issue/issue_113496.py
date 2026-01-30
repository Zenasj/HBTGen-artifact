import torch.nn as nn

py
import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed.checkpoint import FileSystemWriter, save_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import StateDictType
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


class BoringModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(5, 5), torch.nn.ReLU(), torch.nn.Linear(5, 1))
        self.pretrained_model = torch.nn.Sequential(
            torch.nn.Linear(5, 5),
            torch.nn.ReLU(),
        )

    def forward(self, x1, x2):
        x2_hidden = self.pretrained_model(x2)
        hiddens = torch.cat([x1, x2_hidden], dim=1)
        return self.model(hiddens)


def main(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "
    dist.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    
    model = BoringModel()
    model = FSDP(model, auto_wrap_policy=ModuleWrapPolicy({torch.nn.Linear}), device_id=rank)

    state_dict_type_context = FSDP.state_dict_type(
        module=model, state_dict_type=StateDictType.SHARDED_STATE_DICT,
    )

    with state_dict_type_context:
        state_dict = model.pretrained_model.state_dict()
    
    writer = FileSystemWriter(path="model.ckpt")
    save_state_dict(state_dict, writer)

    x1 = torch.ones((2, 6, 5))
    x2 = torch.ones((2, 6, 5)) * 2
    _ = model(x1, x2)


if __name__ == "__main__":
    mp.spawn(main, nprocs=2)

self.model.train()
with torch.no_grad():
        _ = model(
            torch.ones((2, 6, 5)), torch.ones((2, 6, 5))
        )

for k, v in module.state_dict().items():
     print(f"{k} = {v.shape}")