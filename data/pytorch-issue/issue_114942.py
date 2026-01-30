import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(*[
            nn.Conv2d(64, 512, 1, stride=1, padding=0, groups=1),
        ])
        self.post = nn.Linear(8, 16)
        return
    
    def forward(self, xs):
        hs = self.net(xs)
        hs = hs.reshape([xs.shape[0], -1, 64, 64, 64]).transpose(1,4)
        hs = self.post(hs)
        return hs

device = "mps"

my_model = Model().to(device)
optimizer = optim.AdamW(my_model.parameters(), lr=1e-4)

batch_size = 2
target = torch.rand(batch_size, 64, 64, 64, 16).to(device)
xs = torch.rand(batch_size, 64, 64, 64).to(device)

pred = my_model(xs)
loss = torch.mean(target - pred)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(loss.item())