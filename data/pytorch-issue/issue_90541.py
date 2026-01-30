import torch
import torch.nn as nn

# Assuming process group has been initialized (e.g. world size of 2)
model = torch.nn.Linear(10, 10)
fsdp_model = FullyShardedDataParallel(model)
inp = torch.randn((2, 10))
fsdp_model(inp)