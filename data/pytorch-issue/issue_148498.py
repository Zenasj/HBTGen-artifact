import torch.nn as nn

from pathlib import Path

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn

model = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10)
)

dist.init_process_group(
    backend="gloo", world_size=1, rank=0, init_method="tcp://localhost:10998"
)

future = dcp.async_save(
    {"model": model.state_dict()},
    checkpoint_id=Path("checkpoint_dir"),
    process_group=dist.new_group(backend="gloo"),
)
future.result()