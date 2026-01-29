# torch.rand(2, 100, device=device) ← Add a comment line at the top with the inferred input shape
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(100, 50)

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    model = MyModel()
    return model

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

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    device = torch.device("cuda", 0)  # Assuming rank 0 for simplicity
    x = torch.rand(2, 100, device=device)
    return x

def work(rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "1234"
    dist.init_process_group("nccl", world_size=2, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    model = my_model_function().to(device)
    model = FullyShardedDataParallel(model)
    x = GetInput()

    y = model(x)

    from torch.distributed.checkpoint import save
    with get_sharded_state_dict_context(model):
        state = {"model": model.state_dict()}
    save(state, checkpoint_id="fsdp_model.pt")

def run():
    mp.spawn(work, nprocs=2)

if __name__ == "__main__":
    run()

