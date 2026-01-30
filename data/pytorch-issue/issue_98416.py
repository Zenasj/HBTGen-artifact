#!/usr/bin/env python3

import time
import torch
import torch._dynamo as dynamo
import torchvision.models as models

model = models.alexnet()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
compiled_model = torch.compile(model, dynamic=True)

x = torch.randn(16, 3, 224, 224)
optimizer.zero_grad()

epoches=10
count = []
for epoch in range(epoches):
    start = time.time()

    #out = model(x)
    out = compiled_model(x)
    out.sum().backward()
    optimizer.step()

    end = time.time()
    count.append(end - start)
    print(f"Epoch {epoch}/{epoches} time: {end - start}")

print(f"Epoch avg time: {sum(count)/len(count)}")