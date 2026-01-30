import torch.nn as nn

torch.load(save_file, map_location=0)

torch.load(save_file, map_location=1)

import torch

def save_dummy(path):
    torch.save([torch.nn.Parameter(torch.randn(1000, 1000))], path)

def load_saved(path, map_location):
    restored_var = torch.load(path, map_location)
    return restored_var

save_dummy('foo.pt')
restored = load_saved('foo.pt', map_location=torch.device(1))

import torch

def save_dummy(path):
    torch.save([torch.nn.Parameter(torch.randn(1000, 1000))], path)

def load_saved(path, map_location):
    restored_var = torch.load(path, map_location)
    return restored_var

save_dummy('foo.pt')
restored = load_saved('foo.pt', map_location=torch.device(0))

import torch

def save_dummy(path):
    torch.save([torch.nn.Parameter(torch.randn(1000, 1000))], path)

def load_saved(path, map_location):
    restored_var = torch.load(path, map_location)
    return restored_var

save_dummy('foo.pt')
restored = load_saved('foo.pt', map_location='cuda1')