import torch.nn as nn

import torch 

input_tensor = torch.ones((1,1,512,512))

sparse = torch.sparse_coo_tensor(size=(1,10,512,512))

dense  = torch.zeros((1,10,512,512))

print(input_tensor.mul(dense).size())

print(input_tensor.mul(sparse).size())

import torch
import torch.nn.functional as F

input_tensor = torch.ones((1,1,512,512), requires_grad=True)
input_tensor.clamp(min=0)

theta = 3.14159

optimiser  = torch.optim.Adam([input_tensor], lr=1e-4)
loss_function = torch.nn.MSELoss()

with torch.no_grad():
    filtering = torch.nn.Conv1d(1,1,4,4)
    torch.nn.init.constant_(filtering.weight, 1)

rotation_tensor = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                [torch.sin(theta), torch.cos(theta), 0]]).unsqueeze(0)

grid = F.affine_grid(rotation_tensor, input_tensor.size())

sparse = torch.sparse_coo_tensor(size=(1,10,512,512))
target_x = torch.ones((1,10))

optimiser.zero_grad()
x = torch.sum(F.grid_sample(input_tensor, grid, padding_mode='zeros').mul(sparse), dim=[2,3])
filtered = filtering(x)
loss = loss_function(filtered,target_x) 
loss.backward()
optimiser.step()

sparse