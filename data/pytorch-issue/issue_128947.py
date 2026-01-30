import torch.nn as nn

import torch
from torch import nn

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

conv = nn.Conv3d(256, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1)).to('cuda:0')

x = torch.rand([1, 256, 4, 1090, 1922], dtype=torch.float32).to('cuda:0')
# OOM when setting dim2 larger than 4.
# x = torch.rand([1, 256, 6, 1090, 1922], dtype=torch.float32).to('cuda:0')

torch.cuda.synchronize()
print(f"allocated: {torch.cuda.memory_allocated() / (1024**3)}, peak: {torch.cuda.max_memory_allocated() / (1024**3)}")
out = conv(x)
print(out.shape)
torch.cuda.synchronize()
print(f"allocated: {torch.cuda.memory_allocated() / (1024**3)}, peak: {torch.cuda.max_memory_allocated() / (1024**3)}")

# allocated: 7.993362903594971, peak: 7.993362903594971
# torch.Size([1, 64, 2, 1088, 1920])
# allocated: 8.98945665359497, peak: 16.98281955718994