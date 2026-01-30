import torch.nn as nn

import torch
from torch import nn


class WeightNormModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(16, 16, 8, padding=4, groups=2)
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        return hidden_states


model = WeightNormModule()
loss_fn = nn.MSELoss()

input_values = torch.ones(2, 16, 4, dtype=torch.float32)
labels = torch.ones(2, 16, 5, dtype=torch.float32)

out = model(input_values)

loss = loss_fn(out, labels)
loss.backward()

import torch
from torch import nn

class WeightNormModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(16, 16, 8, padding=4, groups=2)
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        return hidden_states


model = WeightNormModule()
loss_fn = nn.MSELoss()

input_values = torch.ones(2, 16, 4, dtype=torch.float32)
labels = torch.ones(2, 16, 5, dtype=torch.float32)

out = model(input_values)

loss = loss_fn(out, labels)
loss.backward()

out = model(input_values)

loss = loss_fn(out, labels)
loss.backward()

import torch
from torch import nn

print(torch.__version__)
class WeightNormModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(16, 16, 8, padding=4, groups=2)
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        return hidden_states


model = WeightNormModule()
loss_fn = nn.MSELoss()

input_values = torch.ones(2, 16, 4, dtype=torch.float32)
labels = torch.ones(2, 16, 5, dtype=torch.float32)

for i in range(5):
    out = model(input_values)
    loss = loss_fn(out, labels)
    loss.backward()

with open("/proc/self/maps") as f:
  print(f.read())