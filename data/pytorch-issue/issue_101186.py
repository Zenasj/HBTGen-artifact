import torch

indices = torch.tensor([
    [0, 1, 1, 2, 2, 3],
    [1, 0, 2, 1, 3, 2],
])
values = torch.ones(6)

M = torch.sparse_coo_tensor(indices, values, size=(4, 4))
assert not M.is_coalesced()
M = M.coalesce()
assert M.is_coalesced()

torch.save(M, 'M.pt')
M = torch.load('M.pt')
assert M.is_coalesced()

torch.save((M, M.is_coalesced()), 'M.pt')
M, M_is_coalesced = torch.load('M.pt')
M._coalesced_(M_is_coalesced)