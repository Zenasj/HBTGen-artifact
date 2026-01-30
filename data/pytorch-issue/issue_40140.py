import torch
a = torch.randn(1)
torch.save(a, 'a.pt')

import torch
torch.load('a.pt')

torch.save(a, 'a.pt', _use_new_zipfile_serialization=False)