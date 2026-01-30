import torch.nn as nn

import torch
import torch.nn.functional as F
import torch.optim as optim

class TestModel(torch.nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.rand(dim, dtype=torch.cfloat))
  
  def forward(self, x):
    return torch.abs(x @ self.weight)

N = 100
dim = 5

x = torch.rand((N, dim), dtype=torch.cfloat).to("cuda")
y = torch.rand(N).to("cuda")
model = TestModel(dim).to("cuda")

opt = optim.Adam(model.parameters(), weight_decay=0.2)
pred = model.forward(x)
loss = F.mse_loss(pred, y)
opt.zero_grad()
loss.backward()
opt.step()

device_grads

device_params

# Handle complex parameters
device_grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_grads]
device_exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_exp_avgs]
device_exp_avg_sqs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_exp_avg_sqs]
params_ = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_params]

# update steps
torch._foreach_add_(device_state_steps, 1)

if weight_decay != 0:
    device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)