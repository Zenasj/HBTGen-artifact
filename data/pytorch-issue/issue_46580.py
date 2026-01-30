import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(256, 256, num_layers=2)

    def forward(self, x):
        _, (last_h_t, _) = self.lstm(x)
        return last_h_t


model = Model()
model.to("cuda:0")

example_in = torch.randn(16, 4, 256).cuda()

jit_model = torch.jit.trace(model, example_in)

out = jit_model(example_in)

print("ok")

jit_model.to("cuda:1")
example_in = example_in.to("cuda:1")
out = jit_model(example_in)

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(256, 256, num_layers=2)

    def forward(self, x):
        init_ht = torch.zeros(2, x.size(1), 256).to(x.device)
        init_ct = torch.zeros(2, x.size(1), 256).to(x.device)
        _, (last_h_t, _) = self.lstm(x, (init_ht, init_ct))
        return last_h_t


model = Model()
model.to("cuda:0")

example_in = torch.randn(16, 4, 256).cuda()

jit_model = torch.jit.trace(model, example_in)
torch.jit.save(jit_model, "test.pt")

jit_model.to("cuda:1")
example_in = example_in.to("cuda:1")
out = jit_model(example_in)

class Model(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  lstm : __torch__.torch.nn.modules.rnn.LSTM
  def forward(self: __torch__.Model,
    x: Tensor) -> Tensor:
    _0 = self.lstm
    _1 = ops.prim.NumToTensor(torch.size(x, 1))
    _2 = torch.zeros([2, int(_1), 256], dtype=6, layout=0, device=torch.device("cpu"), pin_memory=False)
    hx = torch.to(_2, dtype=6, layout=0, device=torch.device("cuda:0"), pin_memory=False, non_blocking=False, copy=False, memory_format=None)
    _3 = ops.prim.NumToTensor(torch.size(x, 1))
    _4 = torch.zeros([2, int(_3), 256], dtype=6, layout=0, device=torch.device("cpu"), pin_memory=False)
    hx0 = torch.to(_4, dtype=6, layout=0, device=torch.device("cuda:0"), pin_memory=False, non_blocking=False, copy=False, memory_format=None)
    return (_0).forward(x, hx, hx0, )

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(256, 256, num_layers=2)

    def forward(self, x):
        # init_ht = torch.zeros(2, x.size(1), 256).to(x.device)
        # init_ct = torch.zeros(2, x.size(1), 256).to(x.device)
        init_ht = torch.zeros_like(x)[:2, :, :]
        init_ct = torch.zeros_like(x)[:2, :, :]
        _, (last_h_t, _) = self.lstm(x, (init_ht, init_ct))
        return last_h_t


model = Model()
model.to("cuda:0")

example_in = torch.randn(16, 4, 256).cuda()

jit_model = torch.jit.trace(model, example_in)
torch.jit.save(jit_model, "test.pt")

jit_model.to("cuda:1")
example_in = example_in.to("cuda:1")
out = jit_model(example_in)

class Model(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  lstm : __torch__.torch.nn.modules.rnn.LSTM
  def forward(self: __torch__.Model,
    x: Tensor) -> Tensor:
    _0 = self.lstm
    _1 = torch.zeros_like(x, dtype=6, layout=0, device=torch.device("cuda:0"), pin_memory=False, memory_format=None)
    _2 = torch.slice(torch.slice(_1, 0, 0, 2, 1), 1, 0, 9223372036854775807, 1)
    hx = torch.slice(_2, 2, 0, 9223372036854775807, 1)
    _3 = torch.zeros_like(x, dtype=6, layout=0, device=torch.device("cuda:0"), pin_memory=False, memory_format=None)
    _4 = torch.slice(torch.slice(_3, 0, 0, 2, 1), 1, 0, 9223372036854775807, 1)
    hx0 = torch.slice(_4, 2, 0, 9223372036854775807, 1)
    return (_0).forward(x, hx, hx0, )