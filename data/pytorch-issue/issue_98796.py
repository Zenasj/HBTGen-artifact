# torch.rand(3, 3, dtype=torch.float)  # as a sparse CSR tensor
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, adj_t):
        # Workaround approach using _sparse_csr_sum
        deg2 = torch._sparse_csr_sum(adj_t, dim=1, keepdim=True).to(torch.float)
        # Attempt original approach (may raise error)
        try:
            deg1 = adj_t.sum(dim=1).to(torch.float)
            return torch.allclose(deg1, deg2)
        except Exception:
            return torch.tensor(False, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Create a sparse CSR tensor for testing
    crow_indices = torch.tensor([0, 2, 3, 3])
    col_indices = torch.tensor([0, 1, 2])
    values = torch.tensor([1., 1., 1.])
    adj_t = torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        values,
        size=(3, 3),
        device='cpu'  # Use CPU to avoid device dependency
    )
    return adj_t

