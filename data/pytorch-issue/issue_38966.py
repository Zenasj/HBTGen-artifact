import torch.nn as nn

import torch

model = torch.nn.Sequential(
    torch.nn.Linear(in_features=1746, out_features=500, bias=True),
    torch.nn.RReLU(lower=0.125, upper=0.3333333333333333),
    torch.nn.Linear(in_features=500, out_features=100, bias=True),
    torch.nn.RReLU(lower=0.125, upper=0.3333333333333333),
    torch.nn.Linear(in_features=100, out_features=2, bias=True)
)
dev = torch.device("cuda")
mem = 480593
intrain = torch.rand((int(mem), 1746), device=dev, dtype=torch.float32)
outtrain = torch.rand((int(mem), 2), device=dev, dtype=torch.float32)
model.to(device=dev)
# model.train()
before = torch.cuda.memory_allocated(dev)
try:
    model(intrain)
except:
    # raise
    pass
after = torch.cuda.memory_allocated(dev)
print("Memory:")
print(before, "- before")
print(after, "- after")

after = torch.cuda.memory_allocated(dev)