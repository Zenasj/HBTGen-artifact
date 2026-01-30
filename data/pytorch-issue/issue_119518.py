import torch

class Subclass(torch.Tensor):
    ...

tensor = torch.randn((3,))
subclass = Subclass(tensor)

def get_amax(subclass):
    amax = torch.max(torch.abs(subclass))
    return amax

amax = torch.compile(get_amax)(subclass)