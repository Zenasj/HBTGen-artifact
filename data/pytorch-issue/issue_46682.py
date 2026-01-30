import torch.nn as nn

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        return "a" in "abcd"

m = MyModel()
scripted = torch.jit.script(m)
scripted()

import torch


@torch.jit.script
def substr_check(x: str):
    return x.find("you") != -1


print(substr_check("yo yo check it out"))
print(substr_check("you you check it out"))