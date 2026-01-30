import torch
indices=torch.tensor([[7, 1, 3]])
values=torch.tensor([[1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.]])
x = torch.sparse_coo_tensor(indices, values, size=(10, 3))
values=torch.tensor(1.).expand(3, 3)
y = torch.sparse_coo_tensor(indices, values, size=(10, 3))
z = x + y

# Should have been all 2's in `values`
tensor(indices=tensor([[7, 1, 3]]),
       values=tensor([[2., 1., 1.],
                      [1., 1., 1.],
                      [1., 1., 1.]]),
       size=(10, 3), nnz=3, layout=torch.sparse_coo)