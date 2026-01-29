# torch.rand(20, 200000, 500, dtype=torch.float32).to_sparse()  # Input shape for MyModel

import torch
from functools import reduce
from torch import nn

def ainb(a, b):
    """Gets mask for elements of a in b."""
    mask = torch.zeros_like(a, dtype=torch.bool)
    for elem in b:
        mask |= (a == elem)
    return mask

class MyModel(nn.Module):
    def forward(self, x):
        slices = [10, ":", ":"]  # Slicing pattern from the issue example
        sliced = self.slice_torch_sparse_coo_tensor(x, slices)
        return sliced.to_dense()
    
    def slice_torch_sparse_coo_tensor(self, t, slices):
        new_shape = []
        new_slices = []
        is_range = []
        for idx, s in enumerate(slices):
            if s == ":":
                is_range.append(True)
                new_shape.append(t.shape[idx])
                new_slices.append(torch.tensor(range(t.shape[idx])))
            elif isinstance(s, int):
                is_range.append(False)
                new_slices.append(torch.tensor([s]))
                new_shape.append(1)  # Corrected size for single-element slices
            elif isinstance(s, list):
                is_range.append(True)
                new_slices.append(torch.tensor(s))
                new_shape.append(len(s))
            else:
                raise NotImplementedError(f"Slicing with {s} is not supported")
        
        assert len(slices) == len(t.size())
        
        # Ensure slices are 1D tensors
        for i in range(len(new_slices)):
            if new_slices[i].ndim > 1:
                new_slices[i] = torch.squeeze(new_slices[i])
        
        t = t.coalesce()
        indices = t.indices()
        values = t.values()
        
        for dim, slice_ in enumerate(new_slices):
            if is_range[dim] and indices.shape[1] > 0:
                low = max(torch.min(indices[dim]), torch.argmin(slice_)).item()
                high = torch.max(indices[dim]).item()
                mask = ainb(indices[dim], slice_[low:high+1])
            else:
                mask = ainb(indices[dim], slice_)
            indices = indices[:, mask]
            values = values[mask]
        
        # Adjust indices for integer slices (set to 0 in the dimension)
        for dim in range(len(is_range)):
            if not is_range[dim]:
                indices[dim].fill_(0)  # Set indices to 0 for integer-sliced dimensions
        
        # Handle new_indices construction
        if values.size(0) < reduce(lambda a, b: a * b, new_shape):
            new_indices = indices
        else:
            # Generate full grid for dense-like cases (rare case)
            grid = torch.where(torch.zeros(new_shape) == 0)
            new_indices = torch.stack([g.flatten() for g in grid])
        
        return torch.sparse_coo_tensor(new_indices, values, new_shape).coalesce()

def my_model_function():
    return MyModel()

def GetInput():
    idx = torch.tensor([[10], [500], [1]])
    values = torch.tensor([1.0], dtype=torch.float32)
    shape = (20, 200000, 500)
    return torch.sparse_coo_tensor(idx, values, shape, dtype=torch.float32)

