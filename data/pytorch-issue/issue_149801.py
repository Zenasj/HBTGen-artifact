import torch

def opt_fn(input_tensor, index_tensor, src_tensor):
    return torch.func.vmap(lambda t: torch.scatter_add(t, 0, index_tensor, src_tensor))(input_tensor)

input_tensor = torch.randn(3)
index_tensor = torch.tensor([0, 1, 2])
src_tensor = torch.tensor([1.0, 2.0, 3.0])
opt_fn(input_tensor, index_tensor, src_tensor)

# Traceback (most recent call last):
#   File "/home/guilhermeleobas/git/pytorch/a.py", line 10, in <module>
#     opt_fn(input_tensor, index_tensor, src_tensor)
#     ~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/guilhermeleobas/git/pytorch/a.py", line 5, in opt_fn
#     return torch.func.vmap(lambda t: torch.scatter_add(t, 0, index_tensor, src_tensor))(input_tensor)
#            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
#   File "/home/guilhermeleobas/git/pytorch/torch/_functorch/apis.py", line 202, in wrapped
#     return vmap_impl(
#         func, in_dims, out_dims, randomness, chunk_size, *args, **kwargs
#     )
#   File "/home/guilhermeleobas/git/pytorch/torch/_functorch/vmap.py", line 334, in vmap_impl
#     return _flat_vmap(
#         func,
#     ...<6 lines>...
#         **kwargs,
#     )
#   File "/home/guilhermeleobas/git/pytorch/torch/_functorch/vmap.py", line 484, in _flat_vmap
#     batched_outputs = func(*batched_inputs, **kwargs)
#   File "/home/guilhermeleobas/git/pytorch/a.py", line 5, in <lambda>
#     return torch.func.vmap(lambda t: torch.scatter_add(t, 0, index_tensor, src_tensor))(input_tensor)
#                                      ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# RuntimeError: index 1 is out of bounds for dimension 1 with size 1