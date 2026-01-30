import torch.nn as nn

import torch
from typing import Dict
from torch import nn

@torch.jit.script
class Batch:
    def __init__(self, tensor):
        self.src = tensor

class UDF(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, edges: Batch) -> Dict[str, torch.Tensor]:
        return {'h': edges.src}
    
a = Batch(torch.arange(100))

def ff(eb: Batch):
    udf = UDF()
    return udf(eb)

sf = torch.jit.script(ff)

import torch
from typing import Dict
from torch import nn

@torch.jit.script
class Batch:
    def __init__(self, tensor):
        self.src = tensor

class BatchWrapper(nn.Module):
    def __init__(self, batch):
        super().__init__()
        self.batch = batch
    

class UDF(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, edges: Batch) -> Dict[str, torch.Tensor]:
        return {'h': edges.src}


class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.udf = UDF()
        self.batch = Batch(torch.arange(100))
    
    def forward(self):
        return self.udf(self.batch)


def ff():
    udf = FullModel()
    return udf()

sf = torch.jit.script(ff)