import torch
import torch.distributed as dist
from collections import namedtuple

from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp import wrap
import torch.nn as nn
import os

dist.init_process_group(backend="nccl", world_size=1, rank=0)
print("init")

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1,1)

    def forward(self, x):
        return x

class Wrapper(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, inp):
        out = self.w(inp)
        print(out)
        print(out.image.data)
        return (out.projected_text_embeddings, out.projected_image_embeddings)


m = MyModule()
m = FullyShardedDataParallel(m)
model = Wrapper(m)
model=FullyShardedDataParallel(model)

t = torch.ones(1, device='cuda')
FLAVAOutput = namedtuple(
    "FLAVAOutput",
    [
        "image",
        "image_masked",
        "text",
        "text_masked",
        "multimodal",
        "multimodal_masked",
        "projected_image_embeddings",
        "projected_text_embeddings",
    ],
    defaults=(t,t,t,t,t,t,t,t),
)

inp = FLAVAOutput()
out = model(inp)
print(out)