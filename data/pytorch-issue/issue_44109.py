import torch
import torch.nn as nn

conv = nn.Conv2d(128, 3, kernel_size=1).half().cuda()

test_tensor = torch.rand((1, 128, 4096, 4096), device='cuda', dtype=torch.float16)

with torch.no_grad():
    out_tensor = conv(test_tensor)