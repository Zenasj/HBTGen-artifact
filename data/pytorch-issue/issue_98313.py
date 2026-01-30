import torch.nn as nn
import random

from torchani.neurochem import parse_neurochem_resources, Constants
from torchani.aev import AEVComputer
import torch
import numpy as np


class Model(torch.nn.Module):
   def __init__(self, device):
      super(Model, self).__init__()
      info_file_path='ani-2x_8x.info'
      const_file, _,_,_ = parse_neurochem_resources(info_file_path)
      consts = Constants(const_file)
      self.aev_computer = AEVComputer(**consts)
      self.aev_computer.to(device)

   def forward(self, species, positions):
      incoords = positions
      inspecies = species
      aev = self.aev_computer((inspecies.unsqueeze(0), incoords.unsqueeze(0)))
      sumaevs = torch.mean(aev.aevs)

      return sumaevs

## setup
N=100
species = torch.randint(0, 7, (N,), device="cuda")
pos = np.random.random((N, 3))

for optimize in [True, False]: 
   print("JIT optimize = ", optimize)

   torch._C._jit_set_profiling_executor(optimize)
   torch._C._jit_set_profiling_mode(optimize)

   model = Model("cuda")
   model = torch.jit.script(model)

   grads=[]
   for i in range(10):
      incoords = torch.tensor(pos, dtype=torch.float32, requires_grad=True, device="cuda")
      result = model(species, incoords)
      result.backward(retain_graph=True)
      grad = incoords.grad
      grads.append(grad.cpu().numpy())
      print(i,"max percentage error: ",np.max(100.0*np.abs((grads[0]-grads[-1])/grads[0])))