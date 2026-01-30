import torch
import torch.utils._pytree as pytree

# Copied from `test/functorch/common_utils.py`.
def loop(op, in_dims, out_dim, batch_size, *batched_args, **kwarg_values):
    outs = []
    out_spec = None
    for idx in range(batch_size):
        flat_args, args_spec = pytree.tree_flatten(batched_args)
        flat_dims, dims_spec = pytree.tree_flatten(in_dims)
        assert(args_spec == dims_spec)
        new_args = [a.select(in_dim, idx) if in_dim is not None else a for a, in_dim in zip(flat_args, flat_dims)]
        out = op(*pytree.tree_unflatten(new_args, args_spec), **kwarg_values)
        flat_out, out_spec = pytree.tree_flatten(out)
        outs.append(flat_out)

    outs = zip(*outs)
    result = [torch.stack(out_lst) for out_lst in outs]
    return pytree.tree_unflatten(result, out_spec)

boundaries = torch.tensor([[1, 4, 5, 7, 9], [1, 2, 5, 7, 9]])
v = torch.tensor(3)

# Output: tensor([1, 2])
print(loop(torch.searchsorted, (0, None), 0, 2, boundaries, v))  

# IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
print(torch.func.vmap(torch.searchsorted, in_dims=(0, None))(boundaries, v))