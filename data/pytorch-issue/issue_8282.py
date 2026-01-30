import torch
import torch.nn as nn

path = './test_model.pth'

model = nn.Conv2d(64, 1, 3, 1, 1)
torch.save(model.state_dict(), path)

model = nn.Conv2d(64, 32, 3, 1, 1)
model.load_state_dict(torch.load(path))

for w in model.weight[:, :, 0, 0]:
    print(w)