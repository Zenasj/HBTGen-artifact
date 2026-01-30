import torch.nn as nn
import torchvision

import torch
torch.cuda.memory._record_memory_history(
    True, trace_alloc_max_entries=100000, trace_alloc_record_context=True
)

import pickle
import torch
from torchvision.models import resnet18

torch.cuda.memory._record_memory_history(
    True, trace_alloc_max_entries=100000, trace_alloc_record_context=True
)

model = resnet18().cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

input = torch.rand(8, 3, 224, 224, device="cuda")
labels = torch.zeros(8, dtype=torch.long, device="cuda")

model.train()

outputs = model(input)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()

snapshot = torch.cuda.memory._snapshot()

with open(f"snapshot.pickle", "wb") as f:
    pickle.dump(snapshot, f)