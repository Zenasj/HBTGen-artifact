import torch
import importlib
import os

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "6782"
torch.distributed.init_process_group("nccl", rank=0, world_size=1)

mod = importlib.import_module("torchbenchmark.models.Background_Matting")
m = mod.Model(test="train", device="cuda")

m.netG = FSDP(m.netG, use_orig_params=True)

m.invoke()