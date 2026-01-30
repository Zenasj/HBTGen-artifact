import torch.nn as nn

import gc
import os
import warnings
import torch
import torch.distributed as dist


from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy
)
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict

import torch.nn.functional as F


local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)


dist.init_process_group()


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32_000, 1024)
        self.linear = torch.nn.Sequential(*[torch.nn.Linear(1024, 1024) for _ in range(50)])
        self.lm_head = torch.nn.Linear(1024, 32_000)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.linear:
            x = layer(x)
        x = self.lm_head(x)
        return x

model = FakeTransformer().cuda()

model = FSDP(module=model, mixed_precision=MixedPrecision(param_dtype=torch.bfloat16))#, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


MICRO_BS = 1


def save(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        states = {"model": get_model_state_dict(model)}
        dcp.save(state_dict=states, checkpoint_id="checkpoint")
    
    torch.cuda.empty_cache()
    gc.collect()

for i in range(5):
    batch = torch.randint(0, 32_000, (MICRO_BS, 2048)).to(torch.cuda.current_device())
    logits = model(batch)
    
    logits = logits.view(-1, logits.size(-1))
    targets = batch.view(-1)
    loss = F.cross_entropy(logits, targets) # fake target

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i == 2 or i == 4:
        if local_rank == 0:
            print("save")
        save(model)

    cuda_info = torch.cuda.memory_stats("cuda")
    max_active = cuda_info["active_bytes.all.peak"]
    max_reserved = cuda_info["reserved_bytes.all.peak"]

    if local_rank == 0:
        print(f"Step {i+1}: max_active: {max_active/1e9:.2f} GiB, max_reserved: {max_reserved/1e9:.2f} GiB")

dist.destroy_process_group()