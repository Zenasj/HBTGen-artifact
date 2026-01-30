import torch.nn as nn

import torch
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
from torch.distributed._tensor import Shard, distribute_tensor, Replicate
from torch.distributed import _functional_collectives as funcol

mesh = DeviceMesh("cuda", torch.arange(2))

class Test(torch.nn.Module):
  def __init__(self):
    super(Test, self).__init__()
    self.weight1 = torch.nn.Linear(in_features=128, out_features=128)
  def forward(self, input):
    return self.weight1(input)


class Network(torch.nn.Module):
  def __init__(self, mesh):
    super(Network, self).__init__()
    self.linear = Test()

    self.linear_parallelized = parallelize_module(self.linear, mesh, {"weight1": ColwiseParallel()})
    self.linear_compiled = torch.compile(self.linear_parallelized)

  def forward(self, input):
    out = self.linear_compiled(input)
    return out


network = Network(mesh)
network.linear.register_forward_hook(lambda _module, _input, output: funcol.all_gather_tensor(output, -1, mesh))

input = torch.rand([128, 128], requires_grad=True)
input = distribute_tensor(input, mesh, [Replicate()])
output = network(input)

x = torch.unsqueeze(output, -1)
labels = torch.unsqueeze(torch.randint(0, 128, (128,)), -1).cuda()

loss_func = torch.nn.CrossEntropyLoss()
loss = loss_func(x, labels)
loss.backward()