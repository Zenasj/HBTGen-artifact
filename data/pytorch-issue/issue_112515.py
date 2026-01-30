import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.multiprocessing as mp

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.checkpoint.optimizer import (
    load_sharded_optimizer_state_dict,
)

CHECKPOINT_DIR = "scratch/checkpoint/py"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8, 8, bias=False)
        self.fc2 = nn.Linear(8, 8, bias=False)
        self.fc3 = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


def opt_at(opt, idx):
    return list(opt.state.values())[idx]


def init_model(dry_run=True):
    model = Net()
    model = FSDP(model.cuda(dist.get_rank()), ignored_modules=[model.fc1])
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    if dry_run:
        optim.zero_grad()
        model(torch.rand(8, 8)).sum().backward()
        optim.step()
    return model, optim


def train(model, optim):
    for _ in range(10):
        optim.zero_grad()
        model(torch.rand(8, 8)).sum().backward()
        optim.step()


def print_params(stage, model_1, model_2, optim_1, optim_2):
    # with FSDP.summon_full_params(model_1):
    #     with FSDP.summon_full_params(model_2):
    #         print(
    #             f"{stage} --- rank: {dist.get_rank()}\n"
    #             f"model.weight: {model_1.weight}\n"
    #             f"model_2.weight:{model_2.weight}\n"
    #             f"model.bias: {model_1.bias}\n"
    #             f"model_2.bias: {model_2.bias}\n"
    #         )

    # print(
    #     f"{stage} --- rank: {dist.get_rank()}\n"
    #     f"optim exp_avg:{opt_at(optim_1, 0)['exp_avg']}\n"
    #     f"optim_2 exp_avg:{opt_at(optim_2, 0)['exp_avg']}\n"
    #     f"optim exp_avg_sq:{opt_at(optim_1, 0)['exp_avg_sq']}\n"
    #     f"optim_2 exp_avg_sq:{opt_at(optim_2, 0)['exp_avg_sq']}\n"
    # )
    pass


def run_fsdp_checkpoint_example(rank, world_size):
    # Set up world pg
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Create a model
    model_1, optim_1 = init_model()

    # Save the model to CHECKPOINT_DIR
    with FSDP.state_dict_type(model_1, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model_1.state_dict(),
            "optim": FSDP.optim_state_dict(model_1, optim_1),
        }

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
        )

    # Create a second model
    model_2, optim_2 = init_model()

    # Print the model parameters for both models.
    # Before loading, the parameters should be different.
    print_params("Before loading", model_1, model_2, optim_1, optim_2)

    # Load model_2 with parameters saved in CHECKPOINT_DIR
    with FSDP.state_dict_type(model_2, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model_2.state_dict(),
            # cannot load the optimizer state_dict together with the model state_dict
        }

        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
        )
        model_2.load_state_dict(state_dict["model"])

        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=state_dict["model"],
            optimizer_key="optim",
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
        )

        flattened_osd = FSDP.optim_state_dict_to_load(
            model_2, optim_2, optim_state["optim"]
        )
        optim_2.load_state_dict(flattened_osd)

    # Print the model parameters for both models.
    # After loading, the parameters should be the same.
    print_params("After loading", model_1, model_2, optim_1, optim_2)

    train(model_2, optim_2)

    # Shut down world pg
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running fsdp checkpoint example on {world_size} devices.")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    mp.spawn(
        run_fsdp_checkpoint_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )