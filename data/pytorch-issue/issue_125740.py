import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from typing import Callable
from torch.distributed import checkpoint as dist_cp
from torch.distributed.checkpoint.planner import LoadItemType
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy
from torch.optim import SGD

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = nn.Linear(3, 3)
        self.linear_1 = nn.Linear(3, 3)
    def forward(self, x):
        return torch.sum(self.linear_1(self.linear_0(x)))

class MyReader(dist_cp.FileSystemReader):
    def read_data(self, plan, planner):
        need_other_rank_file = False
        for read_item in plan.items:
            relative_file_path = self.storage_data[read_item.storage_index].relative_path
            rank = torch.distributed.get_rank()
            if read_item.type == LoadItemType.TENSOR:
                if (rank == 0 and "__1_0" in relative_file_path) or (rank == 1 and "__0_0" in relative_file_path):
                    print(f"bigning debug rank: {torch.distributed.get_rank()}, path: {relative_file_path}, {read_item}")
                    need_other_rank_file = True
        if need_other_rank_file:
            pass
            #raise RuntimeError("Why rank 0 needs '__1_0.distcp' ?")
        return super().read_data(plan, planner)

class MySGD(SGD):
    def __init__(
        self,
        params,
        lr,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ):
        super().__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = torch.zeros((), dtype=torch.float, device=p.device)
                state['step'] += 1
        super().step(closure)
    
def main(rank, world_size):
    ## INITIALIZE DIST
    # Running on one node so master_addr is just local host
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "28000"
    # All ranks simulataneously init the process group together.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    save_path = "./checkpoint"

    model = MyModel().to(f"cuda:{rank}")
    optimizer = MySGD(model.parameters(), lr=0.01)
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
    )
    optimizer.zero_grad()

    x = torch.rand(([2, 3]), device=f"cuda:{rank}")
    y = fsdp_model(x)
    y.backward()
    optimizer.step()

    # save model
    model_state_dict, opt_state_dict = get_state_dict(
        fsdp_model, 
        optimizer, 
        options=StateDictOptions(full_state_dict=False),
    )
    state_dict = {
        "model": model_state_dict,
        "optimizer": opt_state_dict,
    }
    dist_cp.save(
        state_dict=state_dict,
        storage_writer=dist_cp.FileSystemWriter(save_path)
    )
    print(f"bigning debug saving done")

    # load
    dist_cp.load(
        state_dict=state_dict,
        storage_reader=MyReader(save_path),
    )

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2 
    mp.spawn(
        main,
        args = (world_size, ),
        nprocs=world_size,
        join=True
    )