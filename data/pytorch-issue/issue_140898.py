import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.checkpoint import StorageWriter

from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions
)
CHECKPOINT_DIR = "checkpoint"


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net2 = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self.net2(x)

class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def state_dict(self) -> None:
        return get_model_state_dict(self.model)

    def load_state_dict(self, state_dict) -> None:
        set_model_state_dict(self.model, state_dict)


class OptimizerWrapper(Stateful):
    def __init__(self, model: nn.Module, optim: torch.optim.Optimizer) -> None:
        self.model = model
        self.optim = optim

    def state_dict(self) -> None:
        return get_optimizer_state_dict(self.model, self.optim, options=StateDictOptions(flatten_optimizer_state_dict=True))

    def load_state_dict(self, state_dict) -> None:
        set_optimizer_state_dict(self.model, self.optim, optim_state_dict=state_dict, options=StateDictOptions(flatten_optimizer_state_dict=True))

if __name__ == "__main__":
    rank = 0
    world_size = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "

    dist.init_process_group("cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = ToyModel().to(rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    loss = model(torch.rand(1, device="cuda")).sum()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    # optimizer.zero_grad()  # uncomment me to fix DCP!

    state_dict = {"model": ModelWrapper(model), "optimizer": OptimizerWrapper(model, optimizer)}
    dcp.save(state_dict, checkpoint_id=f"{CHECKPOINT_DIR}")

    with torch.no_grad():
        print("optimizer", optimizer.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=20)
        model.net2.weight.zero_()
        print("optimizer is cleared:", optimizer.state_dict())
        print("model is cleared:", model.state_dict())
        state_dict = {"model": ModelWrapper(model), "optimizer": OptimizerWrapper(model, optimizer)}
        dcp.load(state_dict, checkpoint_id=f"{CHECKPOINT_DIR}")
        print("optimizer after load:", optimizer.state_dict())
        print("model after load:", model.net2.weight)
        dist.destroy_process_group()