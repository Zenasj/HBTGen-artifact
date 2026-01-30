import torch.nn as nn

import torch

model = torch.nn.Linear(4, 4)
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20])

# print(scheduler.state_dict())

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
}, "./checkpoint.pth")
print("SAVE")

torch.load("./checkpoint.pth", weights_only=True)
print("LOAD")