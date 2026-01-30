import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(inputs):
#         some comment
        return inputs

M_instance = M()
torch.jit.script(M_instance)