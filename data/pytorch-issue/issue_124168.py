import torch

tensor_1 = torch.tensor([[3], [2], [1]])

tensor_sparse_1 = torch.sparse_coo_tensor(indices=tensor_1, values=[1], size=[4, 2, 1])

tensor_dense = tensor_sparse_1.to_dense()

print(tensor_dense)

tensor_1 = torch.tensor([[3], [2], [1]])
tensor_sparse_1 = torch.sparse_coo_tensor(indices=tensor_1, values=[1], size=[4, 3, 1], check_invariants=True)

from functools import reduce
import torch


def ainb(a,b):
    indices = torch.zeros_like(a, dtype=torch.uint8)
    for elem in b:
        indices = indices | (a == elem)
    return indices.type(torch.bool) 


def slice_torch_sparse_coo_tensor(t, slices):
    new_shape = []
    new_slices = []
    is_range = []

    for idx, s in enumerate(slices):
        if s == ':':
            is_range.append(True)
            new_shape.append(t.shape[idx])
            new_slices.append(torch.tensor(range(t.shape[idx])))

        elif isinstance(s, int):
            is_range.append(False)
            sl = torch.tensor([s])
            new_slices.append(sl)
            new_shape.append(sl.shape[-1])

        elif isinstance(s, list):
            is_range.append(True)
            sl = torch.tensor(s)
            new_slices.append(sl)
            new_shape.append(sl.shape[-1])

        else:
            raise NotImplementedError(f'Slicing with{s} is not supported')

    assert len(slices) == len(t.size())

    for i, slice in enumerate(new_slices):
        if len(new_slices[i].shape) > 1:
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

    if values.size(0) < reduce(lambda a, b: a * b, new_shape):
        new_indices = indices

    else:
        new_indices = [t[None,:] for t in torch.isfinite(input=-3969248092)]
        new_indices = torch.concat(new_indices, dim=0)

    return torch.sparse_coo_tensor(new_indices, values, new_shape, check_invariants=True).coalesce()

tensor_1 = torch.tensor([[3], [2], [1]])
tensor_sparse_1 = torch.sparse_coo_tensor(indices=tensor_1, values=[1], size=[4, 3, 1])
slice_1 = slice_torch_sparse_coo_tensor(tensor_sparse_1, [2, ':', ':'])
shape = slice_1.shape
cat_empty = torch.empty([0, shape[1], shape[2]], layout=torch.sparse_coo)

res = torch.concat([cat_empty, slice_1], dim=0).to_dense()
print(res)