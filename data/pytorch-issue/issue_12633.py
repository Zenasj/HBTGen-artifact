clone()

coalesce()

coalesce()

clone()

dense.to_sparse()

coalesce()

is_coalesced

add

import torch

def make_sparse_from_indices(indices, vals):
    print("indices.stride: ", indices.stride())
    print("vals.stride: ", vals.stride())
    sparse = torch.sparse_coo_tensor(indices, vals, (3, 3, 3))
    return sparse

inds1 = torch.tensor([0, 1, 0, 0]).view(2, 2).cuda()
inds2 = inds1.t().contiguous().t().cuda()

vals1 = torch.tensor([3, 0, 0, 9, 0, 0]).view(2, 3).cuda()
vals2 = vals1

sparse1 = make_sparse_from_indices(inds1, vals1)
sparse2 = make_sparse_from_indices(inds2, vals2)

z1 = torch.zeros(3, 3, 3).cuda()
print(z1.add(sparse1))
print(z1.add(sparse2))