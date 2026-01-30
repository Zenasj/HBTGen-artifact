import torch

def transpose_inplace_index_add(x, a, index):
    y = x.permute(1, 0)
    y.index_add_(0, index, a)
    return y

x = torch.ones(3, 5)
to_be_added = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])

ref_res = transpose_inplace_index_add(x, to_be_added, index)
print('eager_x.stride: ', x.stride())
print('eager_x: ', x)
print('eager_res.stride: ', ref_res.stride())
print('eager_res: ', ref_res)

x = torch.ones(3, 5)
compiled_fn = torch.compile(transpose_inplace_index_add)
res = compiled_fn(x, to_be_added, index)
print('inductor_x.strdides:', x.stride())
print('inductor_x: ', x)
print('inductor_res.strdides:', res.stride())
print('inductor_res: ', res)