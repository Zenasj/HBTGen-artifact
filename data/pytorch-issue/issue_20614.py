import torch

# Calling resize_ on non-requires-grad value tensor
i2 = torch.zeros([1, 1])
v2 = torch.ones([1, 2, 3])
t2 = torch.sparse_coo_tensor(i2, v2, torch.Size([2, 2, 3]))
v2.resize_(4, 5)
t2.coalesce().values().size()
# On current master, this throws "indices and values must have same nnz, but got nnz from indices: 1, nnz from values: 4", because resizing the original value tensor affects `values_` of the sparse tensor.
# After this PR, this prints "torch.Size([1, 2, 3])", which means resizing the original value tensor doesn't affect `values_` of the sparse tensor.