patched_model = ...
ep = torch.export.export(patched_model, (args,))
torch.export.save(ep, path)  # success
torch.export.load(path)  # failed. Note that this is immediately after torch.export.save (in the same context)

import torch
torch.export.load('seg.pt2')