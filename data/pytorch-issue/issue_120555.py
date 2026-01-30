import torch.nn as nn

py
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def loss_function():
    # Dummy loss calculation
    return torch.ones(1, requires_grad=True)

model = torch.nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)

# Dummy training loop
for epoch in range(10):
    for _ in range(100):
        optimizer.zero_grad()
        loss = loss_function()
        loss.backward()
        optimizer.step()
    
    validation_loss = loss_function().item()
    scheduler.step(validation_loss)
    
    print(scheduler.get_last_lr()) # <<< function doesn't exist
    # works: print(optimizer.param_groups[0]['lr'])