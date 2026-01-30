import torch.nn as nn

import torch


class Net(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.li = torch.nn.Linear(10, 10)

  def forward(self, x):
    y = x + torch.randn(1).to(x.device)
    return self.li(y)

device = "cpu"

net = Net().to(device)

inputs = torch.randn(2, 10).to(device)
traced_model = torch.jit.trace(net, inputs, check_trace=False)
ret1 = traced_model(inputs)
traced_model.save('traced_model.cpt')

device = 'cuda'
model = torch.jit.load('traced_model.cpt').to(device)
inputs = inputs.to(device)
ret2 = model(inputs)