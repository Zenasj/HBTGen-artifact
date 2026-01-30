import torch.nn as nn

import torch
import torch.optim as optim
import torchdynamo

torchdynamo.config.fake_tensor_propagation = True

input = torch.ones([10, 10])
model = torch.nn.Sequential(*[torch.nn.Linear(10, 10) for _ in range(2)])
opt_model = torchdynamo.optimize("aot_nop")(model)
opt_model(input).sum().backward()

optimizer = optim.SGD(opt_model.parameters(), lr=5e-5)
opt_fn = torchdynamo.optimize("aot_nop")(optimizer.step)
opt_fn()

import torch
import torch.optim as optim

input = torch.ones([10, 10])
model = torch.nn.Sequential(*[torch.nn.Linear(10, 10) for _ in range(2)])
opt_model = torch.compile(model)
opt_model(input).sum().backward()

optimizer = optim.SGD(opt_model.parameters(), lr=5e-5)
opt_fn = torch.compile(optimizer.step)
opt_fn()