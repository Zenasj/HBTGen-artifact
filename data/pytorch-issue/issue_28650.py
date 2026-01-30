import torch


idxs = torch.tensor([[0, 2, 3], [1, 1, 2], [2, 1, 4], [3, 5, 1]], device=0)
values = torch.randn((4, 6, 5), device=0, requires_grad=True)

sparse_tensor = torch.sparse_coo_tensor(indices=idxs.t(),
                                        values=values[idxs.split(split_size=1, dim=1)].squeeze(),
                                        size=values.shape)
dense_tensor = torch.sparse.sum(sparse_tensor, dim=2).to_dense()
dense_tensor = dense_tensor.sum(dim=1) # + dense_tensor.sum(dim=1)

(dense_tensor * 1).sum().backward()  #  `view size is not compatible with input tensor's size and stride`
# dense_tensor.sum().backward()  #  no exceptions observed