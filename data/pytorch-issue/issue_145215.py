import torch
 
# Create a sparse tensor
# Input (Sparse Tensor):
# [[0, 1],
#  [1, 0]]
indices = torch.tensor([[0, 1], [1, 0]])
values = torch.tensor([1, 1], dtype=torch.float32)
size = torch.Size([2, 2])
 
sparse_tensor = torch.sparse_coo_tensor(indices, values, size)
 
# Convert sparse tensor to dense tensor
dense_tensor = sparse_tensor.to_dense()
 
# Expected Output (Dense Tensor):
# [[0, 1],
#  [1, 0]]
print("\nDense Tensor:")
print(dense_tensor)

# [[0, 11],
#  [10, 0]]