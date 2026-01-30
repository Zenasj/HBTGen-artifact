import torch.nn as nn

import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
from cyclopts import App
import hashlib

def setup():
    # initialize the process group
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def cleanup():
    dist.destroy_process_group()

def save_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR):
    """Save the model and optimizer state to a checkpoint folder

    Args:
        checkpoint_path: the path to the checkpoint folder
        model: the model to save
        optimizer: the optimizer to save
        scheduler: the scheduler to save
    """

    # 1. Save distributed states
    fs_storage_writer = dcp.FileSystemWriter(checkpoint_path)

    model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
    dcp_state_dict = {
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
    }
    dcp.save(dcp_state_dict, storage_writer=fs_storage_writer)

    # 2. Save global states
    global_state_dict = {
        "scheduler": scheduler.state_dict(),
    }
    torch.save(global_state_dict, os.path.join(checkpoint_path, "global_state_dict.pt"))

def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR) -> dict:
    """Load the model and optimizer state from a checkpoint folder
    
    Args:
        checkpoint_path: the path to the checkpoint folder
        model: the model to load
        optimizer: the optimizer to load
        scheduler: the scheduler to load
    """
    # 1. Load distributed states
    fs_storage_reader = dcp.FileSystemReader(checkpoint_path)

    model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
    dcp_state_dict = {
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
    }
    dcp.load(dcp_state_dict, storage_reader=fs_storage_reader)
    set_state_dict(model, optimizer, model_state_dict=model_state_dict, optim_state_dict=optimizer_state_dict)
    
    # 2. Load global states
    global_state_dict = torch.load(os.path.join(checkpoint_path, "global_state_dict.pt"))
    scheduler.load_state_dict(global_state_dict["scheduler"])

def _round_str(x: float):
    return f"{x:.4f}"

def _round_flatten(a: torch.Tensor, max_size: int = 1000):
    bounds = int(max_size**0.5)
    if len(a.shape) == 1:
        return ",".join(_round_str(i) for i in a[:bounds].flatten())
    elif len(a.shape) == 2:
        return ",".join(_round_str(i) for i in a[:bounds, :bounds].flatten())

def hash_tensor_content(a: torch.Tensor, max_size: int = 1000) -> str:
    return hashlib.md5(_round_flatten(a, max_size=max_size).encode("utf-8")).hexdigest()


app = App()

@app.default
def main(
    save_path: str | None = None,
    load_path: str | None = None,
    ckpt_interval: int = 5,
    total_steps: int = 10,
    trace_file: str | None = None,
):
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")

    # create a model and move it to GPU with id rank
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(rank)
    model = FSDP(model, use_orig_params=False, sharding_strategy=ShardingStrategy.NO_SHARD)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)
    if load_path is not None:
        load_checkpoint(load_path, model, optimizer, scheduler)

    for i in range(total_steps):
        input = torch.arange(16).reshape(2, 8).to(rank)

        print(f"[Rank {rank}] Step {i}", hash_tensor_content(next(model.parameters()).data), flush=True)
        #optimizer.param_groups[0]["lr"] = scheduler.get_last_lr()[0]
        output = model(input, labels=input)
        print(output.loss)

        if trace_file is not None and rank == 0:
            with open(trace_file, "a") as f:
                f.write(f"Rank {rank} Step {i} {hash_tensor_content(next(model.parameters()).data)}\n")

        optimizer.zero_grad()
        output.loss.backward()
        print(f"LR before step: {optimizer.param_groups[0]['lr']}")
        optimizer.step()
        scheduler.step()

        if save_path is not None and (i + 1) % ckpt_interval == 0:
            save_checkpoint(save_path + f"/step_{i}", model, optimizer, scheduler)

if __name__ == "__main__":
    setup()
    app()
    cleanup()

def _init_state_dict(state_dict: STATE_DICT_TYPE) -> None:
    state_dict_assigned_storage = tree_map_only(
        torch.Tensor, lambda v: _init_meta_tensor(v), state_dict
    )
    # The inplace version of tree_map_only, tree_map_only_ doesn't seem to work.
    # So we need to temporariy update the each element in the state dict with meta tensor.
    for k in state_dict.keys():
        state_dict[k] = state_dict_assigned_storage[k]

def _init_state_dict(state_dict: STATE_DICT_TYPE) -> None:
    tree_map_only_(torch.Tensor, _init_meta_tensor, state_dict)

def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR) -> dict:
   """Load the model and optimizer state from a checkpoint folder
   
   Args:
       checkpoint_path: the path to the checkpoint folder
       model: the model to load
       optimizer: the optimizer to load
       scheduler: the scheduler to load
   """
   # 1. Load distributed states
   fs_storage_reader = dcp.FileSystemReader(checkpoint_path)

   model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
   dcp_state_dict = {
       "model": model_state_dict,
       "optimizer": optimizer_state_dict,
   }
   dcp.load(dcp_state_dict, storage_reader=fs_storage_reader)
   assert optimizer_state_dict["param_groups"][0]["lr"] == dcp_state_dict["optimizer"]["param_groups"][0]["lr"]
   set_state_dict(model, optimizer, model_state_dict=dcp_state_dict["model"], optim_state_dict=dcp_state_dict["optimizer"])
   
   # 2. Load global states
   global_state_dict = torch.load(os.path.join(checkpoint_path, "global_state_dict.pt"))
   scheduler.load_state_dict(global_state_dict["scheduler"])