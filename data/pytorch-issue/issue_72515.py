import torch.nn as nn

self.trace = [torch.zeros_like(p.data, requires_grad=False) for p in self.model.parameters()]
...
loss.backward()
for idx, p in enumerate(self.model.parameters()):
    # self.gamma and self.lamda are scalars
    self.trace[idx] = self.gamma * self.lamda * self.trace[idx] + p.grad
    # delta is a scalar
    p.grad = delta * self.trace[idx]

...
# delta is a scalar
p.grad = delta * self.trace[idx].reshape(p.grad.shape)

eval_gradients = torch.autograd.grad(loss, self.model.parameters())
for idx, p in enumerate(self.model.parameters()):
    self.trace[idx] = self.gamma * self.lamda * self.trace[idx] + eval_gradients[idx]
    p.grad = delta * self.trace[idx]

input = torch.Tensor(10,64)
model = torch.nn.Linear(64,1)
output = model(input).sum()
output.backward()
delta = torch.Tensor(1,1)
for p in model.parameters():
  p.grad = delta * p.grad

import torch
model = torch.nn.Linear(64,1)
model.bias.grad = torch.tensor(1, dtype=float)