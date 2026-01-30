import torch
import torch.nn as nn

with PackageExporter('my_package.pt', verbose=True) as pe:
    pe.extern('__main__') # Where your custom layer is located at
    pe.save_pickle('models', 'model_1.pkl', myModel)

from torch import nn
from torch import tensor

from torch.package import PackageExporter

class Flatten(nn.Module):
    "Flatten `x` to a single dimension, e.g. at end of a model. `full` for rank-1 tensor"
    def __init__(self, full=False): 
        super().__init__()
        self.full = full
    def forward(self, x): return torch.tensor(x.view(-1) if self.full else x.view(x.size(0), -1))

myModel = nn.Sequential(Flatten())

with PackageExporter('my_package.pt', verbose=True) as pe:
    pe.save_pickle('models', 'model_1.pkl', myModel)