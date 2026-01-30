import torch
from torch.export import _trace
            
def f(x):
    return torch.abs(x)

model = _trace._WrapperModule(f)
ep = torch.export.export(model, (torch.randn(8,),))