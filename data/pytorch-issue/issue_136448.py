import torch.nn as nn
import torch 
from torch.nn import functional as F

class PrimIntToTensorModule(torch.nn.Module):
    constant: int

    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self):
        return torch.ops.prim.NumToTensor(self.constant)


constant = 5
model = PrimIntToTensorModule(constant)

ep = torch.export.export(model, ())

print(ep)