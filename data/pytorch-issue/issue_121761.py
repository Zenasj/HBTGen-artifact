import torch.nn as nn

import torch
from torch.export import export

class Sample(torch.nn.Module):
   def __init__(self, size, h_dim, **kwargs):
       super(Sample, self).__init__(**kwargs)
       self.rnn = torch.nn.GRU(size, h_dim, batch_first=True)
 
   def forward(self, x):
       _, states = self.rnn(x)
       return states

mod = Sample(size=19, h_dim=40)
mod.eval()

gm = export(mod, (torch.rand((1, 40, 19)),)).module()
gm=gm.to("cuda")

inputs = (torch.rand((1, 40, 19)).cuda(),)
gm(*inputs)

import torch
from torch.export import export

class Sample(torch.nn.Module):
   def __init__(self, size, h_dim, **kwargs):
       super(Sample, self).__init__(**kwargs)
       self.rnn = torch.nn.GRU(size, h_dim, batch_first=True)
 
   def forward(self, x):
       _, states = self.rnn(x)
       return states

mod = Sample(size=19, h_dim=40)
mod.eval()

gm = export(mod, (torch.rand((1, 40, 19)),)).module()
gm = gm.to("cuda")

for node in gm.graph.nodes:
    if "device" in node.kwargs:
        kwargs = node.kwargs.copy()
        kwargs["device"] = "cuda"
        node.kwargs = kwargs

inputs = (torch.rand((1, 40, 19)).cuda(),)
exported = export(mod, (torch.rand((1, 40, 19)),))

import torch
from torch.export import export

class Sample(torch.nn.Module):
    def __init__(self, size, h_dim, **kwargs):
        super(Sample, self).__init__(**kwargs)
        self.rnn = torch.nn.GRU(size, h_dim, batch_first=True)

    def forward(self, x):
        _, states = self.rnn(x)
        return states

mod = Sample(size=19, h_dim=40)
mod.eval()
ep = export(mod, (torch.rand((1, 40, 19)),))

# Change all the nodes which have "cpu" hard-coded into "cuda"
for node in ep.graph.nodes:
    if "device" in node.kwargs:
        kwargs = node.kwargs.copy()
        kwargs["device"] = "cuda"
        node.kwargs = kwargs

# Move state dict tensors to cuda
for k, v in ep.state_dict.items():
    if isinstance(v, torch.nn.Parameter):
        ep._state_dict[k] = torch.nn.Parameter(v.cuda())
    else:
        ep._state_dict[k] = v.cuda()

# ep is now a "cuda-based" ExportedProgram

gm = ep.module()
gm(torch.rand((1, 40, 19)).cuda())