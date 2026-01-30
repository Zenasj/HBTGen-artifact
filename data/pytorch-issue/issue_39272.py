import torch.nn as nn

py
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

device = "cuda"
model = torch.nn.Linear(512, 4096).to(device)

model = DDP(model, device_ids=[torch.cuda.current_device()])
model._register_comm_hook(state=None, hook=fp16_compress_hook) 
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for _ in range(5):
    y = model(torch.randn(64, 512, device=device)).mean()
    y.backward()
    optimizer.step()
    optimizer.zero_grad()

print(torch.cuda.memory_allocated() / (1024 ** 2))