import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.Linear(2, 2, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.layer(x))

loss_fn = nn.CrossEntropyLoss()
model = Model()
optimizer = torch.optim.AdamW(opt_model.parameters(), lr=0.01)
dummy_data = torch.ones((2, 2))
out = opt_model(dummy_data)

target1 = torch.zeros(2, dtype=torch.long)
optimizer.zero_grad()
loss1 = loss_fn(out, target1)
loss1.backward(retain_graph=True)

optimizer.zero_grad()
target2 = torch.ones(2, dtype=torch.long)
loss2 = loss_fn(out, target2)
loss2.backward()

import functools
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import enable_wrap, transformer_auto_wrap_policy, wrap


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

# TRAINING LOOP
for i in range(10):
    inp = torch.randint(v, (8, 256), device=local_rank)
    label = inp.add(1) % v
    print("Input and labels created")
    optimizer.zero_grad()
    pred = model(inp)
    loss = criterion(pred.view(-1, pred.size(-1)), label.view(-1))
    print("Loss computed. Calling backward() with retain_graph..")
    loss.backward(retain_graph=True)
    print("Gradient computed once. Called backward() again without retain_graph..")
    loss.backward()
    print("Gradient computed twice")
    optimizer.step()
    print("Update step complete")
    if local_rank == 0 and (i + 1) % 10 == 0:
        print(f"Step {i+1}, loss {loss.item()}")