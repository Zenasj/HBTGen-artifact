import torch
from torchdistx import deferred_init

m = deferred_init.deferred_init(lambda: torch.unsqueeze(torch.empty(2), 0))
print(m)

tensor(..., size=(1, 2), fake=True)