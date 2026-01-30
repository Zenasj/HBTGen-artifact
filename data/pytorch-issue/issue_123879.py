import torch

with torch.device('meta'):
    timesteps = torch.tensor([0, 1], dtype=torch.int64)
    storage = timesteps.untyped_storage()
    ssize = storage.size()
    meta = torch.empty((), dtype=torch.int64)
    meta.set_(storage, 0, (), ())
    assert ssize == storage.size()