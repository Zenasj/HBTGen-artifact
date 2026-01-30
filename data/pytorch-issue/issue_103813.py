import torch
import torch.nn as nn

[4, 59, 256, 256]

[4, 256, 256]

loss = criterion(pred, target)

criterion = torch.nn.CrossEntropyLoss()
preds = torch.randn((4, 59, 256, 256), dtype=torch.float32)
labels = torch.randn((4, 256, 256), dtype=torch.float32)

loss = criterion(preds, labels)

criterion = torch.nn.CrossEntropyLoss()
preds = torch.randn((4, 59, 256, 256), dtype=torch.float32)
labels = torch.empty((4, 256, 256), dtype=torch.long).random_(59)
loss = criterion(preds, labels)

criterion = torch.nn.CrossEntropyLoss()
preds = torch.randn((4, 59, 256, 256), dtype=torch.float32)
labels = torch.randn((4, 59, 256, 256), dtype=torch.float32).softmax(dim=1)
loss = criterion(preds, labels)