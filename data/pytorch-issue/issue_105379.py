import os

import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.checkpoint import FileSystemReader, load_state_dict, FileSystemWriter, save_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import StateDictType


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 50, bias=False)
        self.l2 = nn.Linear(50, 1, bias=False)


def work(rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "1234"
    dist.init_process_group("nccl", world_size=1, rank=rank)

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    model = MyModel().to(device)
    model = FSDP(model)

    path = "tmp/pytorch_debug_sharded"

    with FSDP.state_dict_type(module=model, state_dict_type=StateDictType.SHARDED_STATE_DICT):
        sd = model.state_dict()

    print(list(sd))
    # Trim off some layers
    del sd["l2.weight"]

    writer = FileSystemWriter(path=path, single_file_per_rank=True)
    save_state_dict(sd, writer)

    reader = FileSystemReader(path=path)
    with FSDP.state_dict_type(module=model, state_dict_type=StateDictType.SHARDED_STATE_DICT):
        holder_state = model.state_dict()
        load_state_dict(holder_state, reader)
        model.load_state_dict(holder_state)

    print("good!")


def run():
    mp.spawn(work, nprocs=1)


if __name__ == "__main__":
    run()