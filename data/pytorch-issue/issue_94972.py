import torch.nn as nn

import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

model = torch.nn.Linear(512, 512).cuda()
inputs = torch.randn(64, 512).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("misc code"):
        for i in range(10):
            model(inputs).sum().backward()

by_operator = prof.key_averages()
data2 = by_operator.table(sort_by="self_cuda_time_total", max_name_column_width=50, row_limit=10)
print(data2)