import torch

with FakeTensorMode.push() as mode:
    b = torch.ones(2)
    c = b.unsqueeze(-1)
    b_ = StorageWeakRef(b.storage())
    c_ = StorageWeakRef(c.storage())
    print(b_.cdata)
    print(c_.cdata)  # their storages are different (now fixed in this PR)