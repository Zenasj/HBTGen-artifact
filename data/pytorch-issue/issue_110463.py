import torch
import torch.nn as nn
inputs = torch.Tensor([[0, 1], [1.0, 0], [0, 1]]).to("mps")
targets = torch.LongTensor([1, 0, 1]).to("mps")
loss = nn.CrossEntropyLoss()
loss(inputs, targets)

import torch
import torch.nn as nn
inputs = torch.Tensor([[0, 1], [1.0, 0], [0, 1]]).to("cpu")
targets = torch.LongTensor([1, 0, 1]).to("cpu")
loss = nn.CrossEntropyLoss()
loss(inputs, targets)

import torch
import torch.nn as nn
inputs = torch.Tensor([[0, 1], [1.0, 0], [0, 1]]).to("mps")
targets = torch.LongTensor([1, 0, 1]).to("mps")
loss = nn.CrossEntropyLoss()
loss(inputs, targets)