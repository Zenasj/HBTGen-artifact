py
class GRUCell(jit.ScriptModule):
  __constants__ = ['hidden_size']

  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.weight = nn.Parameter(torch.randn(input_size + hidden_size, 3 * hidden_size))
    self.bias = nn.Parameter(torch.randn(3 * hidden_size))

  @jit.script_method
  def forward(self, input, hidden):
    update, reset = torch.chunk(torch.sigmoid(torch.addmm(self.bias[:2 * self.hidden_size], torch.cat([input, hidden], dim=1), self.weight[:, :2 * self.hidden_size])), 2, dim=1)
    candidate = torch.tanh(torch.addmm(self.bias[2 * self.hidden_size:], torch.cat([input, reset * hidden], dim=1), self.weight[:, 2 * self.hidden_size:]))
    return update * hidden + (1 - update) * candidate

import torch
import torch.jit as jit
import torch.nn as nn

class GRUCell(jit.ScriptModule):
  __constants__ = ['hidden_size']

  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.weight = nn.Parameter(torch.randn(input_size + hidden_size, 3 * hidden_size))
    self.bias = nn.Parameter(torch.randn(3 * hidden_size))

  @jit.script_method
  def forward(self, input, hidden):
    update, reset = torch.chunk(torch.sigmoid(torch.addmm(self.bias[:2 * self.hidden_size], torch.cat([input, hidden], dim=1), self.weight[:, :2 * self.hidden_size])), 2, dim=1)
    candidate = torch.tanh(torch.addmm(self.bias[2 * self.hidden_size:], torch.cat([input, reset * hidden], dim=1), self.weight[:, 2 * self.hidden_size:]))
    return update * hidden + (1 - update) * candidate



m = GRUCell(1024,1024).cuda()
input = torch.randn(128, 1024, device="cuda").requires_grad_(True)
hidden = torch.randn(128, 1024, device="cuda").requires_grad_(True)
out = m(input, hidden)
gO = torch.rand_like(out)
out.backward(gO)
torch.cuda.synchronize()