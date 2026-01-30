import torch.nn as nn

import torch

#model = torch.nn.TransformerEncoderLayer(1024, 16, 4096, dropout=0.1).cuda()
model = torch.nn.Linear(1024, 1024, bias=False).cuda()

print("after init: memory_allocated:", torch.cuda.memory_allocated() / 2 ** 20)
inputs = torch.rand(1, 512, 1024).cuda()

outputs = model(inputs)
print("after forward: memory_allocated:", torch.cuda.memory_allocated() / 2 ** 20)
print("after forward: max_memory_allocated", torch.cuda.max_memory_allocated() / 2 ** 20)