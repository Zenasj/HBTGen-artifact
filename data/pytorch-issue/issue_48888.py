import random

import torch
import torch.nn as nn
import numpy as np

dtype = torch.float
device = torch.device("cpu")

class MLP(nn.Module):
    def __init__(self, inshape, outshape):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(inshape, outshape)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

data = torch.from_numpy(np.random.rand(1, 10).astype(np.float32))
target = torch.from_numpy(np.random.rand(1, 1).astype(np.float32))

model = MLP(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

predict = model(data)
loss = loss_fn(predict, target)

model.zero_grad()
loss.backward()
optimizer.step()

print("Program done, command prompt expected below.")

if sys.platform == "win32":
    os.system(f'wmic process where processid="{os.getpid()}" call terminate >nul')