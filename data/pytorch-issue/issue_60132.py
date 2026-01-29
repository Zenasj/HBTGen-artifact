# torch.rand(2, 2)
import torch
from torch.nn import Module
from torch.nn.modules.lazy import LazyModuleMixin

class MyModel(LazyModuleMixin, Module):
    def __init__(self):
        super().__init__()

    def initialize_parameters(self, input):
        return None

    def forward(self, input):
        return input

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 2)

