import numpy as np

import functools
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import enable_wrap, transformer_auto_wrap_policy, wrap

savepath = "/data/output/davis-1b/TEMP/"

local_rank = int(os.getenv("LOCAL_RANK"))

torch.manual_seed(42 + local_rank)
torch.cuda.manual_seed_all(42 + local_rank)

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

wrapping_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={})

model_sharding_strategy = ShardingStrategy.FULL_SHARD

torch.cuda.set_device(int(os.getenv("RANK")))
dist.init_process_group("nccl")

v = 2048
d = 1024

model = nn.Sequential(nn.Embedding(v, d), nn.Linear(d, v))

model = FSDP(
    model,
    auto_wrap_policy=wrapping_policy,
    mixed_precision=mp_policy,
    sharding_strategy=model_sharding_strategy,
    device_id=local_rank,
    limit_all_gathers=True,
    use_orig_params=True,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# THIS TRIGGERS THE FUTURE CHECKPOINT LOAD FAILURE
inp = torch.arange(256, device=local_rank).unsqueeze(0)
model.eval()
out = model(inp)
model.train()

# TRAINING LOOP
for i in range(100):
    inp = torch.randint(v, (8, 256), device=local_rank)
    label = inp.add(1) % v
    optimizer.zero_grad()
    pred = model(inp)
    loss = criterion(pred.view(-1, pred.size(-1)), label.view(-1))
    loss.backward()
    optimizer.step()
    if local_rank == 0 and (i + 1) % 10 == 0:
        print(f"Step {i+1}, loss {loss.item()}")

# SAVE CHECKPOINT
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    model_state = model.state_dict()
    writer = FileSystemWriter(savepath, single_file_per_rank=True)
    save_state_dict({"model_state": model_state}, storage_writer=writer, planner=DefaultSavePlanner())

# BUILD SECOND MODEL
model2 = nn.Sequential(nn.Embedding(v, d), nn.Linear(d, v))
model2 = FSDP(
    model2,
    auto_wrap_policy=wrapping_policy,
    mixed_precision=mp_policy,
    sharding_strategy=model_sharding_strategy,
    device_id=local_rank,
    limit_all_gathers=True,
    use_orig_params=True,
)

# LOAD CHECKPOINT INTO MODEL 2
with FSDP.state_dict_type(model2, StateDictType.SHARDED_STATE_DICT):
    state_dict = model2.state_dict()
    state_ckp = {"model_state": state_dict}
    load_state_dict(
        state_dict=state_ckp,
        storage_reader=FileSystemReader(savepath),
        planner=DefaultLoadPlanner(),
    )
    model2.load_state_dict(state_ckp["model_state"])
    model2.to(local_rank)

# COMPARE OUTPUTS
n = 10
probe = torch.arange(n, device=local_rank).unsqueeze(0)
p = probe.reshape(-1).tolist()
model.train()
model2.train()
out1 = model(probe)[0, :n, :6].reshape(-1).tolist()
out2 = model2(probe)[0, :n, :6].reshape(-1).tolist()
if local_rank == 0:
    for i in range(n):
        print("Key:", p[i])
        print("Model 1:", out1[i * 6 : i * 6 + 6])
        print("Model 2:", out2[i * 6 : i * 6 + 6])
        print()