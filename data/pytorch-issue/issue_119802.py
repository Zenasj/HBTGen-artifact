import os
import torch.cuda
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel

def get_sharded_state_dict_context(module):
    from torch.distributed.fsdp.api import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType

    state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
    optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=True)
    state_dict_type_context = FullyShardedDataParallel.state_dict_type(
        module=module,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=state_dict_config,
        optim_state_dict_config=optim_state_dict_config,
    )
    return state_dict_type_context  # type: ignore[return-value]

def work(rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "1234"
    dist.init_process_group("nccl", world_size=2, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    model = nn.Linear(100, 50).to(device)
    model = FullyShardedDataParallel(model)
    x = torch.rand(2, 100, device=device)

    y = model(x)

    from torch.distributed.checkpoint import save
    with get_sharded_state_dict_context(model):
        state = {"model": model.state_dict()}
    save(state, checkpoint_id="fsdp_model.pt")

def run():
    mp.spawn(work, nprocs=2)

if __name__ == "__main__":
    run()