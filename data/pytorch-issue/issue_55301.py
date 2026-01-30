import torch
import torch.nn as nn
import torch.cuda.amp as amp

device = torch.device('cuda')
dtype = torch.float32
memory_format = torch.channels_last
# memory_format = torch.contiguous_format

model1 = nn.Conv2d(3,3,1,1).to(device=device, dtype=dtype, non_blocking=True, memory_format=memory_format)
model2 = nn.Conv2d(3,3,1,1).to(device=device, dtype=dtype, non_blocking=True, memory_format=memory_format)

input = torch.randn(1,3,4,4).to(device, dtype=dtype, memory_format=memory_format)

with amp.autocast():
    out1 = model1(input)
    out2 = model2(out1.detach())

out1 = model1(input)
out1 = out1.contiguous()
out2 = model2(out1.detach())

with torch.no_grad():
    out1_clone = out1.detach().clone()
out2 = model2(out1_clone)