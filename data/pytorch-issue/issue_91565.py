import torch

dev = torch.device('mps')
indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
value = torch.tensor([1, 2, 3])
a = torch.sparse_coo_tensor(indices, value, (4, 4),
                            dtype=torch.float32,
                            device=dev).to_dense()
print(a)