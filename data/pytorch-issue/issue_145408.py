import os

import torch

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1))

    def forward(self, x):
        return self.layer(x)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"

dist.init_process_group()

model = Net()
state_dict = get_model_state_dict(model)
pg = dist.new_group(backend="gloo")

try:
    steps = [10, 20, 30, 40, 50]
    future = None
    for step in steps:
        # simulate a training step, e.g. optimizer updating values
        with torch.no_grad():
            model.weight.data.fill_(step)

        if future is not None:
            future.result()
            future = None
        future = dcp.async_save(
            state_dict,
            checkpoint_id=f"outputs/{step}",
            process_group=pg,
        )

    future.result()

    for step in steps:
        dcp.load(
            state_dict,
            checkpoint_id=f"outputs/{step}",
            process_group=pg,
        )
        assert state_dict["weight"][0, 0] == step, f"got {state_dict['weight'][0, 0]=} on {step=}"
finally:
    dist.destroy_process_group(pg)
    dist.destroy_process_group()