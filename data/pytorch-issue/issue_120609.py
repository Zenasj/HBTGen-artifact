import os
import torch.cuda
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel

def work(rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "1234"
    dist.init_process_group("nccl", world_size=2, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    model = nn.Linear(100, 50).to(device)
    model = FullyShardedDataParallel(model)
    x = torch.rand(2, 100, device=device)
    _ = model(x)

    from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions
    from torch.distributed.checkpoint import save, load

    options = StateDictOptions(full_state_dict=False)
    state = {"model": get_model_state_dict(model, options=options)}
    save(state, checkpoint_id="fsdp_model.pt")
    
    state = {"model": model.state_dict()}
    load(state, checkpoint_id="fsdp_model.pt")
    set_model_state_dict(model, state["model"], options=options)

def run():
    mp.spawn(work, nprocs=2)

if __name__ == "__main__":
    run()