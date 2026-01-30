import torch

from torch.nn import Module

m = Module()
m.state_dict()  # this is fine

m.submodule = Module()
m.state_dict()  # this falsely raises a DeprecationWarning