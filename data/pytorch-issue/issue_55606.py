import torch.nn as nn

import torch
from torch import nn

model = nn.Conv2d(8, 64, 3, padding=1)
input = torch.randn(1, 8, 272, 272)

with torch.autograd.profiler.profile(record_shapes=True, with_flops=True) as prof:
    with torch.autograd.profiler.record_function("model_inference"):
        model(input)

events = prof.key_averages(group_by_input_shape=True)
print(events.table())
print(events.table())