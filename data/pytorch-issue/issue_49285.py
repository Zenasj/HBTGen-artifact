import torch.nn as nn

import torch
params = torch.nn.ParameterList([torch.nn.Parameter(torch.ones([1])) for _ in range(2)])

self.training = True

import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")