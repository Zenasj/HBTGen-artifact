import torch
for attribute in dir(torch):
    if getattr(torch, attribute).__class__ == torch.dtype:
        __all__.append(attribute)