import torch.nn as nn

import time
import torch


conv_layer = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dtype=torch.float16)
input_tensor = torch.rand([16, 256, 512, 512]).to(conv_layer.weight.dtype) - 0.5
with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
    with torch.no_grad():
        for i in range(2):
            start = time.time()
            out = conv_layer(input_tensor)
            end = time.time()
            print(f"time costs: {end-start} s")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))