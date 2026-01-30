from functools import reduce

import torch

def ainb(a,b):
    """gets mask for elements of a in b"""
    indices = torch.zeros_like(a, dtype = torch.uint8)
    for elem in b:
        indices = indices | (a == elem)

    return indices.type(torch.bool)
def slice_torch_sparse_coo_tensor(t, slices):
    """
    params:
    -------
    t: tensor to slice
    slices: slice for each dimension

    returns:
    --------
    t[slices[0], slices[1], ..., slices[n]]
    """
    new_shape = []
    new_slices  = []
    is_range = []
    for idx,s in enumerate(slices):
        if s == ":":
            is_range.append(True)
            new_shape.append(t.shape[idx])
            new_slices.append(torch.tensor(range(t.shape[idx])))
        elif isinstance(s, int):
            is_range.append(False)
            sl = torch.tensor([s])
            new_slices.append(sl)
            new_shape.append(sl.shape[-1])
        elif isinstance( s, list):
            is_range.append(True)
            sl = torch.tensor(s)
            new_slices.append(sl)
            new_shape.append(sl.shape[-1])
        else:
            raise NotImplementedError(f"Slicing with{s} is not supported")


    assert len(slices) == len(t.size())
    for i, slice in enumerate(new_slices):
            if len(new_slices[i].shape) >1:
                new_slices[i] = torch.squeeze(new_slices[i])

    t = t.coalesce()
    indices = t.indices()
    values = t.values()
    for dim, slice in enumerate(new_slices):
        if is_range[dim] and indices[dim].shape[-1]>0:
            low = max(torch.min(indices[dim]),torch.argmin(slice)).item()
            high = torch.max(indices[dim]).item()
            mask = ainb(indices[dim], slice[low:high+1])
        else:
            mask = ainb(indices[dim], slice)
        indices = indices[:, mask]
        values = values[mask]
    if(values.size(0) < reduce(lambda a,b: a*b,new_shape)):
        new_indices = indices
    else:
        new_indices = [t[None,:] for t in torch.where(torch.zeros(new_shape) == 0)]
        new_indices = torch.concat(new_indices, dim=0)
    return torch.sparse_coo_tensor(new_indices, values, new_shape).coalesce()

idx = torch.tensor([[10], [500], [1]])
sparse_m = torch.sparse_coo_tensor(idx, torch.tensor([1]), [20, 200000, 500])
slice = slice_torch_sparse_coo_tensor(sparse_m, [10, ":", ":"])
# res = slice.to_dense() #crashes
shape = slice.shape
wired_sigsev_fix = torch.empty([0, shape[1], shape[2]], layout=torch.sparse_coo)
res = torch.concat([wired_sigsev_fix,slice], dim=0).to_dense()  # doesn't crash in debugger 
print(res)
assert res.shape[0] == slice.shape[0]
assert res.shape[1] == slice.shape[1]
assert res.shape[2] == slice.shape[2]

def slice_torch_sparse_coo_tensor(t, slices):
    ...
    return torch.sparse_coo_tensor(new_indices, values, new_shape, check_invariants=True).coalesce()