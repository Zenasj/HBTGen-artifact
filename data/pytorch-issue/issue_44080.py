import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

N, D_in, H, D_out = 64, 1000, 100, 10

model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, D_out),
)

optimizer = Adam(model.parameters(), lr=1e-04)
scheduler = MultiStepLR(optimizer, milestones=[20,30], gamma=0.5, verbose=True)