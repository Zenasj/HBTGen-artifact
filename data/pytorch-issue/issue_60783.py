import torch.nn as nn

import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
scores = torch.rand(16,8,512,512).half().cuda()
with torch.profiler.profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=False) as prof:
     with record_function("auto_dtype"):
         for i in range(100):
             attn_weights = nn.functional.softmax(scores.float(), dim=-1, dtype=scores.dtype)
print(prof.key_averages().table(sort_by="cuda_time_total"))