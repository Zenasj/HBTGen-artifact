# torch.rand(4, 6, 5, dtype=torch.float32, device=0)  # Inferred input shape from the example's 'values' tensor

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.idxs = torch.tensor(
            [[0, 2, 3], [1, 1, 2], [2, 1, 4], [3, 5, 1]],
            device=0
        )  # Original indices from the issue (shape 4x3)

    def forward(self, values):
        # Split indices into individual dimensions' index tensors
        splits = self.idxs.split(split_size=1, dim=1)
        indices_list = [s.squeeze() for s in splits]  # Three tensors of shape (4,)

        # Extract sparse values using the indices
        sparse_values = values[indices_list[0], indices_list[1], indices_list[2]]  # Index into values tensor

        # Create COO sparse tensor with the indices and extracted values
        sparse_tensor = torch.sparse_coo_tensor(
            indices=self.idxs.t(),  # Transposed to (3,4) for COO format
            values=sparse_values,
            size=values.shape
        )

        # Perform the operations described in the issue
        dense = torch.sparse.sum(sparse_tensor, dim=2).to_dense()
        dense = dense.sum(dim=1)
        return (dense * 1).sum()  # Multiply by 1 to trigger the view-related error

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random tensor matching the input shape (4,6,5)
    return torch.rand((4, 6, 5), dtype=torch.float32, device=0, requires_grad=True)

