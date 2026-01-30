import os
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
import torch.multiprocessing as mp
import torch.nn as nn
from typing import Callable
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.fsdp.wrap import CustomPolicy
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor, distribute_module, DTensor
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.checkpoint.state_dict import (StateDictOptions, get_model_state_dict)

fs_dim = 8
tp_dim = 4
input_dim = 2

class TPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn1 = nn.Linear(fs_dim, tp_dim, bias=False)
    def forward(self, x):
        return self.ffn1(x)
    def fsdp_wrap_fn(self, fsdp_mesh):
        return {'device_mesh': fsdp_mesh}

class FullShardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(input_dim, fs_dim, bias=False)
        self.net2 = nn.Linear(fs_dim, fs_dim, bias=False)
        self.ffn = TPModel()
    def forward(self, x):
        return self.ffn(self.net2(self.net1(x)))

def setup(rank, world_size):
    # Running on one node so master_addr is just local host
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "28000"
    # All ranks simulataneously init the process group together.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
    
def run_fsdp_checkpoint_save_example(rank, world_size, mesh_shape):
    setup(rank, world_size)
    print("set up!")
    mesh_2d = init_device_mesh(
        'cuda', mesh_shape, mesh_dim_names=("dp", "tp")
    )
    tp_mesh = mesh_2d["tp"]
    ffn_fsdp_mesh = mesh_2d["dp"]
    mesh_1d = init_device_mesh('cuda', (world_size,))
    model = FullShardModel().to(rank)
    model.ffn.ffn1.weight = torch.nn.Parameter(distribute_tensor(model.ffn.ffn1.weight, tp_mesh, [Shard(0)]))

    def tensor_to_dtensor_hook(mod, args):
        inp, = args
        return DTensor.from_local(inp, tp_mesh, [Shard(0)])
        # return distribute_tensor(inp, tp_mesh, [Shard(0)])
    
    model.ffn.register_forward_pre_hook(tensor_to_dtensor_hook)
    
    def dtensor_to_tensor_hook(mod, inp, outp):
        return outp.to_local()
    model.ffn.register_forward_hook(dtensor_to_tensor_hook)
    
    def lambda_fn(module: torch.nn.Module):
        ret = False
        if hasattr(module, '_fsdp_wrap'):
            ret = bool(module._fsdp_wrap)
        elif hasattr(module, 'fsdp_wrap_fn') and isinstance(module.fsdp_wrap_fn, Callable):
            ret = module.fsdp_wrap_fn(ffn_fsdp_mesh)
        return ret
    model = FSDP(
        model,
        #sharding_strategy=ShardingStrategy.FULL_SHARD,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        device_mesh=mesh_1d,
        use_orig_params=True,
        auto_wrap_policy=CustomPolicy(lambda_fn),
    )


    if torch.distributed.get_rank() == 3:
        print(model)

    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    optim.zero_grad()


    outp = model(torch.rand(4,input_dim,device="cuda")).sum(dim=1)

    loss = loss_fn(outp, torch.ones(4, device='cuda'))
    loss.backward()
    optim.step()


    cleanup()


if __name__ == "__main__":
    mesh_shape = (2, 2)
    world_size = mesh_shape[0]* mesh_shape[1]
    mp.spawn(
        run_fsdp_checkpoint_save_example,
        args = (world_size, mesh_shape),
        nprocs=world_size,
        join=True
    )