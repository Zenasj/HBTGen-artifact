import torch
ifm = torch.tensor([2])
ifm = ifm.pin_memory()
print(ifm.is_pinned())