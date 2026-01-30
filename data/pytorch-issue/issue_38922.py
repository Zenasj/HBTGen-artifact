import torch
x = torch.full((40, 900, 1), 0, dtype=torch.float32)
x.argmax(dim=2)

import torch
x = torch.full((40, 900, 1), 0, dtype=torch.float32)
x.numpy().argmax(axis=2)

x = torch.full((36, 900, 1), 0, dtype=torch.float32)
x.argmax(dim=2)