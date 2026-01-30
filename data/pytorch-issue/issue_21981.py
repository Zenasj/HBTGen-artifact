import torch
import torchvision
import torch.optim as optim
model = torchvision.models.alexnet(pretrained=False)
optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=0.5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, threshold=0.01, verbose=True)

isinstance(scheduler, optim.lr_scheduler._LRScheduler)
# `out: False`

isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)
# `out: True`